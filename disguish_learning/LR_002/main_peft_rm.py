#encoding uft-8
import os
import sys
sys.path.insert(0, '/root/autodl-tmp/generativeModel/')
os.chdir('/root/autodl-tmp/generativeModel/disguish_learning/LR_001_CausalLM')
import transformers
from transformers import AutoTokenizer,AutoModel
from disguish_learning.models.rm_pair_wise import rm_pair
import torch
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from datasets import load_dataset,load_from_disk
from disguish_learning.utils.data_fn import build_tokenzie_func_pair,data_collator_self


cache_dir = '/root/autodl-tmp/model/'
##############
# 模型部分 THUDM/glm-large-chinese 733m THUDM/glm-10b bigscience/bloom-7b1
base_model_name = 'bigscience/bloom-3b'
base_model =  AutoModel.from_pretrained(base_model_name,cache_dir=cache_dir)

model = rm_pair(base_model)
LORA_WEIGHTS=None#"/root/autodl-tmp/model/lora-alpaca"

lora_config = LoraConfig(
    r=8,#LORA_R,
    lora_alpha=16,#LORA_ALPHA,
    target_modules=["query_key_value"],#TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model,lora_config)
# 最后一层还是要训练的
model.base_model.model.scorer.weight.requires_grad = True

tokenizer = AutoTokenizer.from_pretrained(base_model_name,cache_dir=cache_dir,trust_remote_code=True)

###############
# 数据部分 cahya/instructions-zh train 76.9k eval2.02k test2.02k
data_cache_dir = '/root/autodl-tmp/data/'
dataset_name = "yitingxie-rlhf-reward-datasets"
dataset = load_from_disk(os.path.join(data_cache_dir,dataset_name))

tokenize_func = build_tokenzie_func_pair(tokenizer,max_length=1024)
train_data = dataset['train'].map(tokenize_func)
test_data = dataset['test'].map(tokenize_func)


##############
# 优化部分
ddp = True if torch.cuda.device_count() > 1 else False
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128 // 1,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir="lora-alpaca/log",
        save_total_limit=3,
        max_steps=-1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    tokenizer=tokenizer,
    data_collator=data_collator_self()#transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.base_model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))
if torch.__version__ >= "2":
    model = torch.compile(model)

##############
# 训练循环

trainer.train()


save_path = cache_dir + "/lora-alpaca/reward_model"
model.disable_adapter()
model.save_pretrained(save_path)
torch.save(model.get_base_model(),os.path.join(save_path,'model.bin'))

print("\n If there's a warning about missing keys above, please disregard :)")