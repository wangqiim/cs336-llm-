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
from cs336_basics.operator import sample
from BPETokenizer import BPETokenizer
from jaxtyping import Float, Int

import logging
# 创建 Logger
logger = logging.getLogger("my_app")
logger.setLevel(logging.DEBUG)

# 文件处理器（记录 DEBUG 及以上级别）
file_handler = logging.FileHandler("infer.log")
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
      loss.backward()
      optimizer.step()
      logger.info(f"cur iter {cur_iter}/{step_count}, loss = {loss:3f}")
      if cur_iter % 10 == 0:
        save_checkpoint(net, optimizer, iteration=cur_iter+1, out=checkpoint_save_path)
         
def main():
    parser = argparse.ArgumentParser(description="infer with prompt")
    # checkpoint
    parser.add_argument("--checkpoint_path", type=str, default="/home/wq/workplace/assignment1-basics/dataset/checkopint", help="optimizer, model, iter checkpoint path")
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
    # prompt
    parser.add_argument("--max_context_length", type=int, default=256, help="transformer head num")
    # train
    parser.add_argument("--device", type=str, default="cuda:0", help="train device")
    args = parser.parse_args()
    logger.info(f"args = {args}")

    tokenizer = get_tokenizer(args.tokenizer_load_path)
    logger.info(f"tokenizer load from {args.tokenizer_load_path} success!")
    

    device = torch.device(args.device)
    model = get_model(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, torch.device(args.device), torch.float32)

    model.to(device)
    model.eval()
    
    start_iter = 0
    
    if args.checkpoint_path is not None:
      start_iter = load_checkpoint(args.checkpoint_path, model, None)
      logger.info(f"load checkpoint from {args.checkpoint_path} finish, start_iter = {start_iter}")
      
    logger.info(f"prepare anything, ready to infer!")
    
    prompt = input("type prompt>:")
    
    inputs: Int[torch.Tensor, "1 seq_length"] = torch.tensor([tokenizer.encode(prompt)], dtype=torch.int32).to(device)
    while inputs.size(-1) < args.max_context_length:
      output: Int[torch.Tensor, "1 seq_length"] = model(inputs)
      sample_id = sample(output[0,-1], temp=0.9, top_p=0.9).item()
      # print(f"sample id: {sample_id}")
      if sample_id == 256:
        break
      inputs = torch.cat([inputs, torch.tensor([[sample_id]], dtype=torch.int32, device=device)], dim=-1)
    # print(f"ids len = {inputs.size(-1)}")
    print(f"answer: {tokenizer.decode(inputs[0].tolist())}")

if __name__ == "__main__":
    main()
