import os
import re
import time
from collections import defaultdict

from typing import BinaryIO, Iterator, Iterable

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

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.special_token = "<|endoftext|>"
        self.special_tokens = special_tokens
        if special_tokens is None:
            self.special_tokens = [self.special_token]
        if self.special_token not in self.special_tokens:
            self.special_tokens.append(self.special_token)
        
        self.vocab_index: dict[bytes, int] = {}
        for index, bs in vocab.items():
            self.vocab_index[bs] = index
        
        self.index_merges: dict[tuple[int, int], int] = {}
        for i in range(len(merges)):
            self.index_merges[self.vocab_index[merges[i][0]], self.vocab_index[merges[i][1]]] = i  
    
    @staticmethod
    def from_files(vocab_merge_filepath, special_tokens=None):
        """
        从 JSON 文件中反序列化出 (vocab, merges)
        Args:
            file_path: 输入文件路径
        Returns:
            元组 (vocab_dict, merges_list)
        """
        import json
        
        with open(vocab_merge_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将 base64 字符串转换回 bytes
        vocab = {int(k): v.encode('latin1') for k, v in data["vocab"].items()}
        merges = [
            (m[0].encode('latin1'), m[1].encode('latin1')) for m in data["merges"]
        ]
        return BPETokenizer(vocab, merges, ["<|endoftext|>"])
        
    def _build_chunks(self, text: str, desired_num_chunks: int) -> list[str]:
        desired_num_chunks = 1
        chunks = []
        chunk_size = len(text) // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = len(text)
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            chunk_text = text[initial_position: initial_position + mini_chunk_size]
            while True:
                # Find the special token in the mini chunk
                found_at = chunk_text.find(self.special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            chunks.append(text[start: end])
        return chunks
    
    def encode(self, text: str) -> list[int]:
        # 1. 将text拆分成chunk（用于多线程处理）
        chunks = self._build_chunks(text, desired_num_chunks=1)
        # print(f"chunks: {chunks}")
        
        # 2. 将每个chunk内拆分成words, 将word依次映射成list[int]
        chunk_ids_list = []
        for chunk in chunks:
            # 2.1. 找到所有分隔词，先将分割词转换完成
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            pattern = "(" + "|".join(escaped_tokens) + ")"
            sentence_list = re.split(pattern, chunk)
            ids_list = []
            for sentence in sentence_list:
                if sentence not in self.special_tokens:
                    words_list = pretokenize(sentence)
                    for word in words_list:
                        token_index_list = [self.vocab_index[bytes([b])] for b in word.encode("utf-8")]
                        # todo 对token_index_list进行Merge
                        while True:
                            min_merge_seq = -1
                            merge_i = -1
                            for i in range(len(token_index_list) - 1):
                                index1, index2 = token_index_list[i], token_index_list[i + 1]
                                merge_seq = self.index_merges.get((index1, index2), -1)
                                if merge_seq != -1 and (min_merge_seq == -1 or merge_seq < min_merge_seq):
                                    merge_i = i
                                    min_merge_seq = merge_seq
                            if min_merge_seq == -1:
                                break
                            index1, index2 = token_index_list[merge_i], token_index_list[merge_i + 1]
                            token_index_list[merge_i] = self.vocab_index[self.vocab[index1] + self.vocab[index2]]
                            del token_index_list[merge_i+1]
                        
                        ids_list += token_index_list
                else:
                    ids_list.append(self.vocab_index[bytes([c for c in sentence.encode("utf-8")])])
                    # print(f"special_token: {sentence}")
            # print(f"ids_list: {ids_list}")
        return ids_list
    
    def decode(self, ids: list[int]) -> str:
        combined_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        # 解码并处理非法字符
        return combined_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

########################################################################################
###########################             test            ################################
########################################################################################

def train_and_save(train_txt_path, train_output_path):
    vocab, merges = train_bpe(train_txt_path, 10000, ['<|endoftext|>'], num_process=20)
    serialize_to_file(vocab, merges, train_output_path)
    # chunk read time = 3.00s
    # [build_words_freqs] map time = 24.87s
    # [build_words_freqs] reduce time = 0.06s
    # construct indices frequency list time = 25.06s
    # merge vocab time = 141.18s

def test_compression(tokenizer, sample_text_path):
    start_time = time.time()
    with open(sample_text_path) as f:
        ids = []
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
            
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
    end_time = time.time()
    
    print(f"file:{sample_text_path}, size = f{file_size} , token_count = {len(ids)}, compression ratio = {file_size / len(ids):.2f}, throughput = {file_size/(end_time-start_time):.2f}byte/s")

def load_test(bpe_params_path: str):
    # load
    tokenizer = BPETokenizer.from_files(bpe_params_path)
    # sample test
    assert "hello<|endoftext|> world!" == tokenizer.decode(tokenizer.encode("hello<|endoftext|> world!"))
    # test_compression_ratio
    test_compression(tokenizer, "tests/fixtures/tinystories_sample_5M.txt")
    test_compression(tokenizer, "dataset/TinyStories-valid.txt")
    test_compression(tokenizer, "tests/fixtures/address.txt")
    test_compression(tokenizer, "tests/fixtures/corpus.en")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="bpe_tokenizer.")
    parser.add_argument("--train",  action='store_true', help="训练 tokenizer")
    parser.add_argument("--train_data_path",  type=str, default="dataset/TinyStories-train.txt", help="训练 tokenizer")
    parser.add_argument("--train_bpe_param_path",  type=str, default="dataset/TinyStories-train-output.txt", help="训练 tokenizer")
    parser.add_argument("--eval",  action='store_true', help="评价测试 tokenizer")
    parser.add_argument("--load_bpe_param_path",  type=str, default="dataset/TinyStories-train-output.txt", help="训练 tokenizer")
    args = parser.parse_args()
    args = parser.parse_args()
    if args.train:
        train_and_save(args.train_data_path, args.train_bpe_param_path)
    if args.eval:
        load_test(args.load_bpe_param_path)