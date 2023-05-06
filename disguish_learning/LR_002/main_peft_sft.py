#encoding uft-8
import os

import transformers
from transformers import  AutoModelForCausalLM,AutoTokenizer
import torch
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from datasets import load_dataset


cache_dir = '/root/autodl-tmp/model/'
data_cache_dir = '/root/autodl-tmp/data/'
##############
# 模型部分 THUDM/glm-large-chinese 733m THUDM/glm-10b bigscience/bloom-7b1
model = AutoModelForCausalLM.from_pretrained('THUDM/glm-2b',cache_dir=cache_dir,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-2b',cache_dir=cache_dir,trust_remote_code=True)
print(model.parameters())
print(model.generation_config)
model = prepare_model_for_int8_training(model)
lora_config = LoraConfig(
    r=8,#LORA_R,
    lora_alpha=16,#LORA_ALPHA,
    target_modules=["query_key_value"],#TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model,lora_config)


###############
# 数据部分 cahya/instructions-zh train 76.9k eval2.02k test2.02k
def split_train_example(text:str):
    answer_prefix = "Assistant:"
    prompt_prefix = "User:"

    answer_start_idx = text.find(answer_prefix)
    if(answer_start_idx > 0):
        # this is an trian data
        answer = text[answer_start_idx + len(answer_prefix):]
        prompt = text[:answer_start_idx].replace(prompt_prefix,"")
    else:
        prompt = text
        answer = None

    return prompt,answer


def build_tokenzie_func(tokenizer, pad_idx=0, max_length=256, pad=True):
    def tokenize(example):
        text = example['text']
        prompt, answer = split_train_example(text)
        prompt_idxs = tokenizer(prompt)
        answer_idxs = tokenizer(answer) if answer is not None else []

        example = {}
        example['input_ids'] = prompt_idxs['input_ids'] + answer_idxs['input_ids']
        example['attention_mask'] = prompt_idxs['attention_mask'] + answer_idxs['attention_mask']
        example['labels'] = [0] * len(prompt_idxs['attention_mask']) + answer_idxs['input_ids']
        example['prompt'] = prompt
        example['answer'] = answer

        if (len(example['input_ids']) < max_length):
            count = max_length - len(example['input_ids'])
            example['input_ids'] = example['input_ids'] + [0] * count
            example['attention_mask'] = example['attention_mask'] + [0] * count
            example['labels'] = example['labels'] + [0] * count

        example['input_ids'] = example['input_ids'][:max_length]
        example['attention_mask'] = example['attention_mask'][:max_length]
        example['labels'] = example['labels'][:max_length]

        return example

    return tokenize


tokenize_func = build_tokenzie_func(tokenizer)
data = load_dataset("cahya/instructions-zh",cache_dir=data_cache_dir)

train_data = data['train'].map(tokenize_func)
val_data = data['validation'].map(tokenize_func)
test_data = data['test'].map(tokenize_func)


##############
# 优化部分
ddp = True if torch.cuda.device_count() > 1 else False
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=128 // 4,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir="lora-alpaca",
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))
if torch.__version__ >= "2":
    model = torch.compile(model)

##############
# 训练循环

trainer.train()


save_path = cache_dir + "/lora-alpaca"
model.save_pretrained(save_path)
torch.save(model.base_model,os.path.join(save_path,'model.bin'))

print("\n If there's a warning about missing keys above, please disregard :)")