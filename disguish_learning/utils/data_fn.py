from dataclasses import dataclass

import transformers
from transformers.data.data_collator import DataCollatorMixin
import torch
from transformers.tokenization_utils import PreTrainedTokenizerBase


###############
# 数据部分 cahya/instructions-zh train 76.9k eval2.02k test2.02k
def split_train_example_for_sft(text:str):
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


def build_tokenzie_func(tokenizer:transformers.PreTrainedTokenizer, max_length=256):
    def tokenize(example):
        text = example['text']
        prompt, answer = split_train_example_for_sft(text)
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

def build_tokenzie_func_pair(tokenizer:transformers.PreTrainedTokenizer):
    def tokenize(example):
        """
        ignore label idx should always be -100
        """
        max_length = tokenizer.max_length
        prompt_text = example['prompt']
        chosen_text = example['chosen']
        rejected_text = example['rejected']

        prompt_idxs = tokenizer(prompt_text)
        chosen_idxs = tokenizer(chosen_text)
        rejected_idxs = tokenizer(rejected_text)

        # positive
        positive =  prompt_idxs['input_ids'] + chosen_idxs['input_ids'] + [tokenizer.cls_token_id]
        att_mask_pos =  prompt_idxs['attention_mask'] + chosen_idxs['attention_mask'] + [1]
        # negtive
        negtive =   prompt_idxs['input_ids'] + rejected_idxs['input_ids'] + [tokenizer.cls_token_id]
        att_mask_neg =  prompt_idxs['attention_mask'] + rejected_idxs['attention_mask'] + [1]


        pos_actual_len = len(positive)
        neg_actual_len = len(negtive)
        # pad and truck
        positive = positive + [tokenizer.pad_token_id] * max(0,max_length - len(positive))
        att_mask_pos = att_mask_pos + [0] * max(0,max_length - len(att_mask_pos) )
        negtive = negtive + [tokenizer.pad_token_id] * max(0,max_length - len(negtive))
        att_mask_neg = att_mask_neg + [0] * max(0,max_length - len(att_mask_neg))
        positive = positive[:max_length]
        att_mask_pos = att_mask_pos[:max_length]
        negtive = negtive[:max_length]
        att_mask_neg = att_mask_neg[:max_length]

        example['pos_actual_len'] = pos_actual_len
        example['neg_actual_len'] = neg_actual_len

        example['positive'] = positive
        example['negtive'] = negtive

        example['att_mask_pos'] = att_mask_pos
        example['att_mask_neg'] = att_mask_neg

        return example
    return tokenize

class data_collator_self(DataCollatorMixin):
    return_tensors: str = "pt"
    tokenizer: PreTrainedTokenizerBase

    def batch_tensor_stack(self, features):
        assert len(features) > 0
        # print(features)
        batch = {
            k: [v]
            for k, v in features[0].items()
        }
        # print(batch)

        for fea in features[1:]:
            # print("-"*100)
            # print(fea)
            for k, v in fea.items():
                batch[k].append(v)

        for k in batch:
            try:
                batch[k] = torch.tensor(batch[k], dtype=torch.long)
            except Exception as e:
                # print(k,e)
                batch[k] = batch[k]
        # print("-"*100)
        # print(batch)
        return batch

    def torch_call(self, features):
        import torch
        # print("--"*100)
        # print(features)

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = self.batch_tensor_stack(
            features)  # [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = no_labels_features
        batch['return_loss'] = True
        if labels is None:
            return batch

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        # print(batch)
        return batch