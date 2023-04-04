#encoding=utf-8
import torch.multiprocessing as mp
from tasks.finetune_llama_fasdp import start
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fsdp')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--model_path', type=str, default='/', metavar='N')


    WORLD_SIZE = torch.cuda.device_count()

    # def start(model,
    #           data_path,
    #           optimizer,scheduler,rank,world_size,epochs=1,
    #           save_folder=None
    #           ):

