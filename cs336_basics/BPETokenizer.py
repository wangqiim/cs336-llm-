import os
import re
import time
from collections import defaultdict

from typing import BinaryIO

from multiprocessing import Pool

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
def pretokenize(text: str) -> list[str]:
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(PAT, text)

def vocab_initialize(
        special_tokens: list[str]
    ) -> tuple[list[bytes], dict[bytes, int]]:
    assert len(special_tokens) == 1 and special_tokens[0] == '<|endoftext|>', f"special_tokens must only <|endoftext|>"
    vocab_index2bytes = []
    vocab_bytes2index = {}
    for i in range(256):
        vocab_index2bytes.append(bytes([i]))
        vocab_bytes2index[bytes([i])] = i
    init_byte_cnt = len(vocab_index2bytes)
    
    for i in range(len(special_tokens)):
        special_token = special_tokens[i]
        vocab_index2bytes.append(bytes([c for c in special_token.encode("utf-8")]))
        vocab_bytes2index[bytes([c for c in special_token.encode("utf-8")])] = i + init_byte_cnt
    return vocab_index2bytes, vocab_bytes2index

def chunks_read(input_path: str, num_process: int) -> list[str]:
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_process, "<|endoftext|>".encode("utf-8"))
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    return chunks

class WordCounter:
    @staticmethod
    def init(special_tokens: list[str]):
        escaped_tokens = [re.escape(token) for token in special_tokens]
        WordCounter.pattern = "|".join(escaped_tokens)
    
    @staticmethod
    def process_chunk(chunk: str) -> dict[str, int]:
        words_freq = {}
        texts = re.split(WordCounter.pattern, chunk)
        for text in texts:
            for word in pretokenize(text):
                words_freq[word] = words_freq.get(word, 0) + 1
        return words_freq
    
    @staticmethod
    def merge_counts(words_freq_list: list[dict[str, int]]) -> dict[str, int]:
        merged_words_req = {}
        for words_freq in words_freq_list:
            for word, freq in words_freq.items():
                merged_words_req[word] = merged_words_req.get(word, 0) + freq
        return merged_words_req

def build_words_freqs(
        special_tokens: list[str],
        chunks: list[str],
        vocab_bytes2index: list[bytes],
        num_process: int
    ) -> list[tuple[list[int], int]]:
    
    WordCounter.init(special_tokens)
    start_time = time.time()
    with Pool(processes=num_process) as pool:  # 显式指定核心数
        words_freq_list = pool.map(WordCounter.process_chunk, chunks)
    print(f"[build_words_freqs] map time = {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    reduced_words_freq = WordCounter.merge_counts(words_freq_list)
    print(f"[build_words_freqs] reduce time = {time.time() - start_time:.2f}s")
    
    indices_freq_list = []
    for word, freq in reduced_words_freq.items():
        token_index_list = [vocab_bytes2index[bytes([b])] for b in word.encode("utf-8")]
        indices_freq_list.append((token_index_list, freq))
    return indices_freq_list

def merge(indices_freq_list: list[tuple[list[bytes], int]], counts: dict[tuple[int], int], index1: int, index2: int, new_index: int):
    tmp_list = [0] * 10 # pre alloc
    for indices, freq in indices_freq_list:
        indices_i = 0
        i = 0
        indices_num = len(indices)
        if len(tmp_list) < indices_num:
            tmp_list.extend([0] * (indices_num - len(tmp_list)))
        while i < indices_num:
            if i + 1 < indices_num and indices[i] == index1 and indices[i + 1] == index2:
                tmp_list[indices_i] = new_index
                i += 2
            else:
                tmp_list[indices_i] = indices[i]
                i += 1
            indices_i += 1

        if i != indices_i:
            # update count
            for old_index1, old_index2 in zip(indices[:-1], indices[1:]):
                counts[(old_index1, old_index2)] -= freq
                if counts[(old_index1, old_index2)] == 0:
                    del counts[(old_index1, old_index2)]

            indices[:] = tmp_list[:indices_i]

            for new_index1, nex_index2 in zip(indices[:-1], indices[1:]):
                counts[(new_index1, nex_index2)] += freq
        
def merge_loop(
        vocab_size: int,
        vocab_index2bytes: list[bytes],
        vocab_bytes2index: dict[bytes, int],
        indices_freq_list: list[tuple[list[int], int]]
    )-> list[tuple[int]]:
    merges = []
    counts = defaultdict(int)
    for i in range(len(indices_freq_list)):
        indices, freq = indices_freq_list[i]
        for index1, index2 in zip(indices[:-1], indices[1:]):
            counts[(index1, index2)] += freq
    while len(vocab_index2bytes) < vocab_size and len(counts) != 0:
        max_count = 0
        chosen_pair = None
        chosen_bytes_pair = None
        # find max count merge byte pair
        for pair, count in counts.items():
            if count > max_count:
                chosen_pair = pair
                max_count = count
                chosen_bytes_pair = vocab_index2bytes[pair[0]], vocab_index2bytes[pair[1]]
            elif count == max_count:
                bytes_pair = vocab_index2bytes[pair[0]], vocab_index2bytes[pair[1]]
                if bytes_pair > chosen_bytes_pair:
                    chosen_pair = pair
                    chosen_bytes_pair = bytes_pair
        # generate new index
        new_index = len(vocab_index2bytes)
        bytes1, bytes2 = vocab_index2bytes[chosen_pair[0]], vocab_index2bytes[chosen_pair[1]]
        new_bytes = bytes1 + bytes2
        vocab_index2bytes.append(new_bytes)
        vocab_bytes2index[new_bytes] = new_index
        merges.append((bytes1, bytes2))
        # update indices_freq_list
        merge(indices_freq_list, counts, chosen_pair[0], chosen_pair[1], new_index)
    return merges
   
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str], # must distinct, assume special_tokens = 
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size > 256 + len(special_tokens), f"vocab_size must greater than {256 + len(special_tokens)}"
    num_process = kwargs.get("num_process", 1)  # 默认值 1
    # vocabulary initialization
    vocab_index2bytes, vocab_bytes2index = vocab_initialize(special_tokens) 
    merges = []

    import time
    # chunk read
    start_time = time.time()
    chunks: list[str] = chunks_read(input_path, num_process)
    print(f"chunk read time = {time.time() - start_time:.2f}s")
    
    # 1. 构建频率表
    start_time = time.time()
    indices_freq_list = build_words_freqs(special_tokens, chunks, vocab_bytes2index, num_process)
    print(f"construct indices frequency list time = {time.time() - start_time:.2f}s")
    
    # 2. vocab
    start_time = time.time()
    merges = merge_loop(vocab_size, vocab_index2bytes, vocab_bytes2index, indices_freq_list)
    print(f"merge vocab time = {time.time() - start_time:.2f}s")
    
    vocab = {}
    for i in range(len(vocab_index2bytes)):
        vocab[i] = vocab_index2bytes[i]
    # print(len(vocab))
    return vocab, merges

def serialize_to_file(
    vocab: dict[int, bytes],
    merges:list[tuple[bytes, bytes]], 
    file_path: str
) -> None:
    """
    将 (vocab, merges) 序列化为 JSON 字符串保存到文件
    Args:
        vocab_merges: 元组 (vocab_dict, merges_list)
        file_path: 输出文件路径
    """
    import json
    
    # 将 bytes 转换为 base64 字符串以便 JSON 序列化
    vocab_serializable = {k: v.decode('latin1') for k, v in vocab.items()}
    merges_serializable = [
        (m[0].decode('latin1'), m[1].decode('latin1')) for m in merges
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"vocab": vocab_serializable, "merges": merges_serializable},
            f,
            indent=2  # 可选：美化输出
        )

if __name__ == "__main__":
    vocab, merges = train_bpe("dataset/TinyStories-train.txt", 10000, ['<|endoftext|>'], num_process=20)
    serialize_to_file(vocab, merges, "dataset/TinyStories-train-output.txt")
    # chunk read time = 3.00s
    # [build_words_freqs] map time = 24.87s
    # [build_words_freqs] reduce time = 0.06s
    # construct indices frequency list time = 25.06s
    # merge vocab time = 141.18s

    # vocab, merges = train_bpe("dataset/TinyStories-valid.txt", 1000, ['<|endoftext|>'], num_process=15)
    # serialize_to_file(vocab, merges, "dataset/TinyStories-valid-output.txt")
    # chunkQ read time = 0.04s
    # [build_words_freqs] map time = 0.25s
    # [build_words_freqs] reduce time = 0.01s
    # construct indices frequency list time = 0.27s
    # merge vocab time = 2.58s