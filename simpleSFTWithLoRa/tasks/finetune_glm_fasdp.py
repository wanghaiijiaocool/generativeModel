#encoding=utf-8
import os
import logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from funcutils.utils import generate_prompt,build_generate_and_tokenize_prompt,build_tokenize
import accelerate
from dataset.TextDataset import TextDataset
from transformers import  BloomForCausalLM,BloomTokenizerFast

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import tqdm
import pprint
import torchsnooper

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
import functools

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()
def load_data(path,tokenizer,rank,batch_size=[],world_size=1,val_size=100,CUTOFF_LEN=256,cuda_kwargs=None,split_rank=True):
    data = load_dataset("json", data_files=path)

    train_val = data["train"].train_test_split(
        test_size=val_size, shuffle=True, seed=42
    )
    train_datao = train_val["train"].shuffle()
    train_data = train_datao


    val_data = train_val["test"]

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    generate_and_tokenize_prompt = build_generate_and_tokenize_prompt(tokenizer,CUTOFF_LEN=CUTOFF_LEN)
    tokenize = build_tokenize(tokenizer,CUTOFF_LEN=CUTOFF_LEN)
    train_data = train_data.map(generate_and_tokenize_prompt)
    val_data = val_data.map(generate_and_tokenize_prompt)

    train_ds = TextDataset(train_data)
    test_ds = TextDataset(val_data)

    train_sampler = DistributedSampler(
        train_ds,rank=rank,shuffle=True,num_replicas=world_size)
    test_sampler = DistributedSampler(
        test_ds,rank=rank,shuffle=False,num_replicas=world_size)
    train_kwargs = {
        "batch_size": batch_size[0] if isinstance(batch_size,list) else batch_size,
        "sampler":train_sampler
    }
    test_args = {
        "batch_size": batch_size[1] if isinstance(batch_size, list) else batch_size,
        "sampler": test_sampler
    }

    train_kwargs.update(cuda_kwargs)
    test_args.update(test_args)
    train_dl = torch.utils.data.DataLoader(train_ds,**train_kwargs)
    test_dl = torch.utils.data.DataLoader(test_ds,**test_args)
    return train_dl,test_dl,train_sampler,test_sampler

def load_model(path_or_name,load_in_8bit=False,device_map='auto'):
    print(f"load model in {path_or_name}|{load_in_8bit}|{device_map}")
    model = BloomForCausalLM.from_pretrained(
        path_or_name,
        load_in_8bit=load_in_8bit
    )
    print(f"load model complete")
    tokenizer = BloomTokenizerFast.from_pretrained(path_or_name)
    return model,tokenizer
def prepare_peft(model,target_modules,lora_rank=8,lora_alpha=1,lora_dropout=0.5):
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    return model
@torchsnooper.snoop()
def train(model,dl,optimizer,rank,epoch,sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    if(sampler is not None):
        sampler.set_epoch(epoch)
    step = 0
    for batch in tqdm.tqdm(dl,desc=f"at rank {rank}"):
        input_ids = batch['input_ids'].to(rank)
        att_mask = batch['attention_mask'].to(rank)
        labels = batch['labels'].to(rank)

        optimizer.zero_grad()
        output = model(input_ids=input_ids,attention_mask=att_mask,labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(input_ids)
        if(rank == 0 and step % 50 == 0):
            print('Train Epoch: {}\t Step {} \tLoss: {:.6f}'.format(epoch, step,ddp_loss[0] / ddp_loss[1]))
        step += 1
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

def save_model(save_path,model):
    # state_dict for FSDP model is only available on Nightlies for now
    states = model.state_dict()
    torch.save(states, save_path)

def start_fsdp(rank,world_size,path_or_name,load_in_8bit,device_map,
          batch_size,data_path,cuda_kwargs,
          epochs=1,
          val_size=1000,
          cutoff_len=256,
          save_folder=None,
          lr=1e-5,
          lr_schedule_gamma=0.7,
          ):
    print("rank",rank)
    setup(rank,world_size=world_size)


    torch.cuda.set_device(rank)

    model, tokenizer = load_model(path_or_name,load_in_8bit,device_map)
    model = FSDP(model, device_id=torch.cuda.current_device(),
                 backward_prefetch=BackwardPrefetch.BACKWARD_POST)

    parameters = list(model.parameters())

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_schedule_gamma)

    train_dl, test_dl, train_sampler, test_sampler = load_data(
        path=data_path,rank=rank, tokenizer=tokenizer,
        batch_size=batch_size, world_size=world_size,
        val_size=val_size, CUTOFF_LEN=cutoff_len, cuda_kwargs=cuda_kwargs
    )

    print(f"{model}")
    print(f"rank:{rank} per-gpu/tpu (sharded) parameter num: {sum(p.numel() for p in parameters)}")
    print(f"\n=== optimizer ===\nrank:{rank} \n{pprint.pformat(optimizer)}\n")


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    for epoch in range(1, epochs + 1):
        train(model=model, rank=rank, dl=train_dl, optimizer=optimizer, epoch=epoch, sampler=train_sampler)
        #test(model, rank, world_size, test_loader)
        scheduler.step()
    init_end_event.record()
    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    if(save_folder is not None):
        dist.barrier()
        if(rank == 0):
            save_model(os.path.join(save_folder,"mode.pth"),model)
    cleanup()

def start_deep_speed(rank,world_size,path_or_name,load_in_8bit,device_map,
          batch_size,data_path,cuda_kwargs,
          epochs=1,
          val_size=1000,
          cutoff_len=256,
          save_folder=None,
          lr=1e-5,
          lr_schedule_gamma=0.7,
          ):
    print("rank",rank)
    setup(rank,world_size=world_size)


    torch.cuda.set_device(rank)

    model, tokenizer = load_model(path_or_name,load_in_8bit,device_map)
    model = FSDP(model, device_id=torch.cuda.current_device(),
                 backward_prefetch=BackwardPrefetch.BACKWARD_POST)

    parameters = list(model.parameters())

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_schedule_gamma)

    train_dl, test_dl, train_sampler, test_sampler = load_data(
        path=data_path,rank=rank, tokenizer=tokenizer,
        batch_size=batch_size, world_size=world_size,
        val_size=val_size, CUTOFF_LEN=cutoff_len, cuda_kwargs=cuda_kwargs
    )

    print(f"{model}")
    print(f"rank:{rank} per-gpu/tpu (sharded) parameter num: {sum(p.numel() for p in parameters)}")
    print(f"\n=== optimizer ===\nrank:{rank} \n{pprint.pformat(optimizer)}\n")


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    for epoch in range(1, epochs + 1):
        train(model=model, rank=rank, dl=train_dl, optimizer=optimizer, epoch=epoch, sampler=train_sampler)
        #test(model, rank, world_size, test_loader)
        scheduler.step()
    init_end_event.record()
    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    if(save_folder is not None):
        dist.barrier()
        if(rank == 0):
            save_model(os.path.join(save_folder,"mode.pth"),model)
    cleanup()


