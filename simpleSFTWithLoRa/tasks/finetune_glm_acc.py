#encoding=utf-8
import os
import logging
import torch

from datasets import load_dataset

from transformers import  BloomForCausalLM,BloomTokenizerFast


import tqdm
import pprint
import torchsnooper

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)



def load_data(path,tokenizer):
    data = load_dataset("json", data_files=path)




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
#@torchsnooper.snoop()
def train(model,dl,optimizer,epoch=0,lr_scheduler=None,accelerator=None,accumulate_step = 1,):
    device = accelerator.device
    model.train()
    ddp_loss = torch.zeros(2).to(device)
    step = 0

    print(f"start train epoch: {epoch} @ deivce: {device}")
    for batch in tqdm.tqdm(dl,desc=f"at rank {device}"):
        input_ids = batch['input_ids'].to(device)
        att_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids,attention_mask=att_mask,labels=labels)
        loss = output.loss / accumulate_step if accumulate_step is not None and accumulate_step > 1 else output.loss
        accelerator.backward(loss)
        if (step + 1) % accumulate_step == 0:
            optimizer.step()
            if(lr_scheduler is not None):
                lr_scheduler.step()
            optimizer.zero_grad()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(input_ids)
        if(device == 0 and step % 50 == 0):
            print('Train Epoch: {}\t Step {} \tLoss: {:.6f}'.format(epoch, step,ddp_loss[0] / ddp_loss[1]))

    all_epoch_loss, all_step = accelerator.gather((ddp_loss[0], torch.tensor(step, device=device)))
    if device == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

def save_model(save_path,model):
    # state_dict for FSDP model is only available on Nightlies for now
    states = model.state_dict()
    torch.save(states, save_path)

def start_deep_speed(world_size,path_or_name,load_in_8bit,
          batch_size,data_path,cuda_kwargs,
          epochs=1,
          val_size=1000,
          cutoff_len=256,
          save_folder=None,
          lr=1e-5,
          lr_schedule_gamma=0.7,
          mixed_precision='bf16',
          accumulate_step=1
          ):
    """
    fdsp with accelerator
    :return:
    """
    from accelerate import Accelerator, DeepSpeedPlugin
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step,
                              deepspeed_plugin=deepspeed_plugin)
    print("device",accelerator.device)


    torch.cuda.set_device(accelerator.device)

    model, tokenizer = load_model(path_or_name,load_in_8bit)


    parameters = list(model.parameters())

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_schedule_gamma)

    train_dl, test_dl, train_sampler, test_sampler = load_data(
        path=data_path,rank=rank, tokenizer=tokenizer,
        batch_size=batch_size, world_size=world_size,
        val_size=val_size, CUTOFF_LEN=cutoff_len, cuda_kwargs=cuda_kwargs
    )

    print(f"{model}")
    print(f"rank:{accelerator.device} per-gpu/tpu (sharded) parameter num: {sum(p.numel() for p in parameters)}")
    print(f"\n=== optimizer ===\nrank:{accelerator.device} \n{pprint.pformat(optimizer)}\n")

    model, optimizer, train_dl, test_dl = accelerator.prepare(model,optimizer,train_dl,test_dl)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    for epoch in range(1, epochs + 1):
        train(model=model, dl=train_dl,optimizer=optimizer, epoch=epoch)
        scheduler.step()
    init_end_event.record()
    if accelerator.device == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    if(save_folder is not None):
        if(accelerator.device == 0):
            save_model(os.path.join(save_folder,"mode.pth"),model)


