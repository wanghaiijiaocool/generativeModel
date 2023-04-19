#encoding=utf-8
from tasks.finetune_glm_acc import start_deep_speed
import argparse
import torch

import os
import logging


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    parser = argparse.ArgumentParser(description='fsdp')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/llama7bhf', metavar='N')
    parser.add_argument('--data_path_or_name', type=str, default='./alpaca_data_cleaned.json', metavar='N')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--load_in_8bit', action="store_true",default=False)
    parser.add_argument('--device_map', type=str,default="balanced")
    parser.add_argument('--cuda_kwargs', type=str, default=None)
    parser.add_argument('--val_size', type=int, default=200)
    parser.add_argument('--cutoff_len',type=int,default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_schedule_gamma', type=float, default=7e-1)
    parser.add_argument('--save_folder', type=str, default=None)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    start_deep_speed(world_size=world_size,
                     path_or_name=args.model_path,


    )




