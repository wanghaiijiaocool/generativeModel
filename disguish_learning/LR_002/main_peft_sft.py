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
model.disable_adapter()
model.save_pretrained(save_path)
torch.save(model.get_base_model(),os.path.join(save_path,'model.bin'))

print("\n If there's a warning about missing keys above, please disregard :)")