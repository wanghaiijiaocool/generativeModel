#encoding=utf-8
from tasks.finetune_bloom_fasdp import start
import argparse
import torch
import torch.multiprocessing as mp
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

    if(args.cuda_kwargs is None):
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': False}
    else:
        cuda_kwargs = eval(args.cuda_kwargs)
    print("world_size",world_size)
    # rank, world_size, path_or_name, load_in_8bit, device_map,
    # batch_size, data_path, cuda_kwargs,
    # epochs = 1,
    # val_size = 1000,
    # cutoff_len = 256,
    # save_folder = None,
    # lr = 1e-5,
    # lr_schedule_gamma = 0.7,
    # start(world_size,args.model_path,args.load_in_8bit,args.device_map,
    #         args.batch_size,args.data_path_or_name,cuda_kwargs,
    #         args.max_epochs,
    #         args.val_size,
    #         args.cutoff_len,
    #         None,
    #         args.lr,
    #         args.lr_schedule_gamma)
    mp.spawn(start
        ,args=(world_size,args.model_path,args.load_in_8bit,None,
            args.batch_size,args.data_path_or_name,cuda_kwargs,
            args.max_epochs,
            args.val_size,
            args.cutoff_len,
            None,
            args.lr,
            args.lr_schedule_gamma)
        ,nprocs=world_size,join=True
    )

