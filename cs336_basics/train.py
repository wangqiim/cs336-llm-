import os
import torch
import argparse
import logging
import random
from cs336_basics import TransformerLM
from cs336_basics import AdamW
from cs336_basics import load_checkpoint, save_checkpoint
from cs336_basics import DataLoader
from cs336_basics import softmax
from cs336_basics import cross_entropy
from BPETokenizer import BPETokenizer

import logging
# 创建 Logger
logger = logging.getLogger("my_app")
logger.setLevel(logging.DEBUG)

# 文件处理器（记录 DEBUG 及以上级别）
file_handler = logging.FileHandler("train.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 控制台处理器（记录 INFO 及以上级别）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 记录日志
logger.debug("调试信息（仅写入文件）")
logger.info("普通信息（文件和控制台都会输出）")

def get_tokenizer(load_path: str):
  return BPETokenizer.from_files(load_path)

def get_model(
        vocab_size: int, 
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device, 
        dtype: torch.dtype
  ):
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device, dtype)
    return model

def get_optimizer(
    model,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float
  ):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr, weight_decay, (beta1, beta2), eps)
    return optimizer

def get_data_loader(np_data_path: str, len: int, batch_size: int, context_length: int, device: str):
    return DataLoader(np_data_path, len, batch_size, context_length, device)

def train_loop(
    start_iter: int,
    step_count: int,
    data_loader: DataLoader,
    context_length: int,
    net: TransformerLM,
    optimizer: AdamW,
    checkpoint_save_path: str
  ):
    net.train()
    for cur_iter in range(start_iter, step_count):
      net.zero_grad()
      inputs, target = data_loader.get_batch()
      logistic = net(inputs)
      loss = 0
      for i in range(context_length):
        loss += cross_entropy(logistic[:, i, ...], target[:, i])
      loss /= context_length
      loss.backward()
      optimizer.step()
      logger.info(f"cur iter {cur_iter}/{step_count}, loss = {loss:3f}")
      if cur_iter % 10 == 0:
        save_checkpoint(net, optimizer, iteration=cur_iter+1, out=checkpoint_save_path)
      if cur_iter % 50 == 0:
        save_checkpoint(net, optimizer, iteration=cur_iter+1, out=f"{checkpoint_save_path}_{cur_iter}")

def generate_np_txt_file(tokenizer: BPETokenizer, raw_txt_path: str, bin_txt_path: str) -> int:
    import numpy as np
    if not os.path.exists(bin_txt_path):
        with open(raw_txt_path) as f:
            ids = [0] * (1922767089//3) # pre alloc
            id_cnt = 0
            for _id in tokenizer.encode_iterable(f):
                ids[id_cnt] = _id
                id_cnt += 1
                if id_cnt % 100000000 == 0:
                    logger.info(f"cur decode ids {id_cnt}")
                    
            
            np_array = np.array(ids[:id_cnt], dtype=int)
            np_array.tofile(bin_txt_path)
            return id_cnt
    else:
        loaded_array = np.fromfile(bin_txt_path, dtype=int)
        return len(loaded_array)
             
def main():
    parser = argparse.ArgumentParser(description="Train a basic model.")
    # checkpoint
    parser.add_argument('--from_scratch', action='store_true', help='从头训练')
    parser.add_argument("--checkpoint_path", type=str, default="/home/wq/workplace/assignment1-basics/dataset/checkopint", help="optimizer, model, iter checkpoint path")
    parser.add_argument("--checkpoint_save_path", type=str, default="/home/wq/workplace/assignment1-basics/dataset/checkopint", help="tokenizer vocab_size")
    # tokenizier 
    parser.add_argument("--vocab_size", type=int, default=10000, help="tokenizer vocab_size")
    parser.add_argument("--tokenizer_load_path", type=str, default="/home/wq/workplace/assignment1-basics/dataset/TinyStories-train-output.txt", help="tokenizer param path")
    # transformer
    parser.add_argument("--context_length", type=int, default=256, help="Number of epochs to train")
    parser.add_argument("--d_model", type=int, default=512, help="transformer d_model")
    parser.add_argument("--d_ff", type=int, default=1344, help="transformer d_ff")
    parser.add_argument("--rope_theta", type=int, default=10000, help="transformer rope theta")
    parser.add_argument("--num_layers", type=int, default=4, help="transformer block num")
    parser.add_argument("--num_heads", type=int, default=16, help="transformer head num")
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="adamW lr")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="adamW weight_decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="adamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="adamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="adamW eps")
    # train
    parser.add_argument("--device", type=str, default="cuda:0", help="train device")
    parser.add_argument("--batch_size", type=int, default=32, help="train batchsize")
    parser.add_argument("--step_count", type=int, default=5000, help="train stap_count")
    # generate dataset np
    parser.add_argument("--raw_txt_path", type=str, default="/home/wq/workplace/assignment1-basics/dataset/TinyStories-train.txt", help="data loader raw file")
    parser.add_argument("--bin_txt_path", type=str, default="/home/wq/workplace/assignment1-basics/dataset/TinyStories-train.bin", help="data loader np file")
    args = parser.parse_args()
    logger.info(f"args = {args}")

    tokenizer = get_tokenizer(args.tokenizer_load_path)
    logger.info(f"tokenizer load from {args.tokenizer_load_path} success!")
    len_ids = generate_np_txt_file(tokenizer, args.raw_txt_path, args.bin_txt_path)
    logger.info(f"train file, ids_num is {len_ids}!")
    
    data_loader = get_data_loader(args.bin_txt_path, len_ids, args.batch_size, args.context_length, args.device)

    device = torch.device(args.device)
    model = get_model(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, torch.device(args.device), torch.float32)
    optimizer = get_optimizer(model, args.lr, args.weight_decay, args.beta1, args.beta2, args.eps)

    model.to(device)
    
    start_iter = 1
    
    if not args.from_scratch and args.checkpoint_path is not None:
      start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
      logger.info(f"load checkpoint from {args.checkpoint_path} finish, start_iter = {start_iter}")
    else:
      logger.info(f"train from scratch!")
    logger.info(f"prepare anything, ready to train!")

    train_loop(
        start_iter,
        args.step_count,
        data_loader,
        args.context_length,
        model,
        optimizer,
        args.checkpoint_save_path)

if __name__ == "__main__":
    main()
