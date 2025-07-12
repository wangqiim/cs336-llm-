import os
import re
from collections import defaultdict

from typing import BinaryIO
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

def chunks_read(input_path: str) -> list[str]:
    chunks = []
    num_process = 1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_process, "<|endoftext|>".encode("utf-8"))
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    return chunks

def build_words_freqs(
        special_tokens: list[str],
        chunks: list[str],
        vocab_bytes2index: list[bytes]
    ) -> list[tuple[list[int], int]]:
    words_freq = {}
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    for chunk in chunks:
        texts = re.split(pattern, chunk)
        for text in texts:
            for word in pretokenize(text):
                # word = word.replace(" ", "")
                words_freq[word] = words_freq.get(word, 0) + 1
                
    indices_freq_list = []
    for word, freq in words_freq.items():
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
    
    # vocabulary initialization
    vocab_index2bytes, vocab_bytes2index = vocab_initialize(special_tokens) 
    merges = []

    # chunk read
    chunks = chunks_read(input_path)
    
    # 1. 构建频率表
    indices_freq_list = build_words_freqs(special_tokens, chunks, vocab_bytes2index)
    
    # 2. vocab
    merges = merge_loop(vocab_size, vocab_index2bytes, vocab_bytes2index, indices_freq_list)
    
    vocab = {}
    for i in range(len(vocab_index2bytes)):
        vocab[i] = vocab_index2bytes[i]
    # print(len(vocab))
    return vocab, merges
            
if __name__ == "__main__":
    train_bpe("/home/wq/workplace/assignment1-basics/tests/fixtures/address.txt", 500, ['<|endoftext|>'])
