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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str], # must distinct, assume special_tokens = 
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size > 256 + len(special_tokens), f"vocab_size must greater than {256 + len(special_tokens)}"
    assert len(special_tokens) == 1 and special_tokens[0] == '<|endoftext|>', f"special_tokens must only <|endoftext|>"
    vocab_index2bytes = []
    vocab_bytes2index = {}
    merges = []
    # vocabulary initialization
    for i in range(256):
        vocab_index2bytes.append(bytes([i]))
        vocab_bytes2index[bytes([i])] = i
    init_byte_cnt = len(vocab_index2bytes)
        
    for i in range(len(special_tokens)):
        special_token = special_tokens[i]
        vocab_index2bytes.append(bytes([c for c in special_token.encode("utf-8")]))
        vocab_bytes2index[bytes([c for c in special_token.encode("utf-8")])] = i + init_byte_cnt

    chunks = []
    num_process = 1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_process, "<|endoftext|>".encode("utf-8"))
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    
    # 1. 构建频率表
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
    
    # 2. merge loop
    merge_cnt = 0
    while len(vocab_index2bytes) < vocab_size:
        counts = defaultdict(int)
        for indices, freq in indices_freq_list:
            for index1, index2 in zip(indices[:-1], indices[1:]):
                counts[(index1, index2)] += freq
        if len(counts) == 0:
            break
        max_count = 0
        chosen_pair = None
        chosen_bytes_pair = None
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
        new_index = len(vocab_index2bytes)
        bytes1, bytes2 = vocab_index2bytes[chosen_pair[0]], vocab_index2bytes[chosen_pair[1]]
        new_bytes = bytes1 + bytes2
        vocab_index2bytes.append(new_bytes)
        vocab_bytes2index[new_bytes] = new_index
        # print(f"merge_cnt = {merge_cnt} ")
        merges.append((bytes1, bytes2))
        # update indices_freq_list
        for indices, freq in indices_freq_list:
            new_indices = []
            i = 0 
            while i < len(indices):
                if i + 1 < len(indices) and indices[i] == chosen_pair[0] and indices[i + 1] == chosen_pair[1]:
                    new_indices.append(new_index)
                    i += 2
                else:
                    new_indices.append(indices[i])
                    i += 1
            indices[:] = new_indices
        merge_cnt += 1
        
    vocab = {}
    for i in range(len(vocab_index2bytes)):
        vocab[i] = vocab_index2bytes[i]
    # print(len(vocab))
    return vocab, merges
            
if __name__ == "__main__":
    train_bpe("/home/wq/workplace/assignment1-basics/tests/fixtures/address.txt", 500, ['<|endoftext|>'])
