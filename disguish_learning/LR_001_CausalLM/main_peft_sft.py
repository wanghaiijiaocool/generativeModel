#encoding=utf-8
#!pwd

import os

os.chdir('/root/autodl-tmp/generativeModel/disguish_learning/LR_001_CausalLM')

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from datasets import load_dataset,DownloadMode,load_from_disk

cache_dir = '/root/autodl-tmp/model/'
model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-3b', cache_dir=cache_dir,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-3b', cache_dir=cache_dir, trust_remote_code=True)

# make model as

#model = prepare_model_for_int8_training(model)
lora_config = LoraConfig(
    r=8,  # LORA_R,
    lora_alpha=16,  # LORA_ALPHA,
    target_modules=["query_key_value"],  # TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    merge_weights=True
)
model = get_peft_model(model, lora_config)

data_cache_dir = '/root/autodl-tmp/data/'
dataset_name = "cahya-instructions-zh"

#data = load_dataset("cahya/instructions-zh",cache_dir=data_cache_dir,max_retries=10,download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS)
data = load_from_disk(os.path.join(data_cache_dir,dataset_name))

for x in data['train']:
    print(x)
    break


def split_train_example(text: str):
    answer_prefix = "Assistant:"
    prompt_prefix = "User:"

    answer_start_idx = text.rfind(answer_prefix)
    if (answer_start_idx > 0):
        # this is an trian data
        answer = text[answer_start_idx + len(answer_prefix):]
        prompt = text[:(answer_start_idx + len(answer_prefix))]  # .replace(prompt_prefix,"")
    else:
        prompt = text
        answer = None

    return prompt, answer


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

train_data = data['train'].map(tokenize_func)


val_data = data['validation'].map(tokenize_func)
test_data = data['test'].map(tokenize_func)

ddp = True if torch.cuda.device_count() > 1 else False

log_path = cache_dir + "/lora-alpaca/log"
if(not os.path.exists(log_path)):
    os.makedirs(log_path,exist_ok=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=128 // 2,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=log_path,
        save_total_limit=3,
        max_steps=-1,
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

print("\n If there's a warning about missing keys above, please disregard :)")
model.disable_adapter()
save_path = cache_dir + "/lora-alpaca/sft_model"
model.save_pretrained(save_path)
torch.save(model.get_base_model(), os.path.join(save_path, 'model.bin'))

# 测试部分



model.disable_adapter()

for n, x in model.base_model.model.named_parameters():
    if ("embedding" in n):
        print(x.size())

prompt = """
User:漫威电影都有哪些？
Assistant:《复仇者联盟》、《钢铁侠》、《美国队长》、《雷神》、《绿巨人》、《黑寡妇》、《蚁人》、《惊奇队长》、《黑豹》、《复仇者联盟2》、《银河护卫队》、《复仇者联盟3》、《复仇者联盟4》
User:《复仇者联盟》讲述了什么？
"""
prompt_idxs = tokenizer(prompt)

prompt_idxs = {
    key: torch.LongTensor([prompt_idxs[key]]).cuda()
    for key in prompt_idxs
}

with torch.no_grad():
    x = model.generate(input_ids=prompt_idxs['input_ids'], max_new_tokens=56)

for z in x.cpu().numpy():
    y = tokenizer.decode(z)
    print(y)


