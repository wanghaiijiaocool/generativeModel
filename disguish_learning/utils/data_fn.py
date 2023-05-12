from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
import torch

def build_tokenzie_func(tokenizer, pad_idx=0, max_length=1024, pad=True):
    def tokenize(example):
        """
        ignore label idx should always be -100
        """
        prompt_text = example['prompt']
        chosen_text = example['chosen']
        rejected_text = example['rejected']

        prompt_idxs = tokenizer(prompt_text)
        chosen_idxs = tokenizer(chosen_text)
        rejected_idxs = tokenizer(rejected_text)

        # positive
        positive = prompt_idxs['input_ids'] + chosen_idxs['input_ids']
        att_mask_pos = prompt_idxs['attention_mask'] + chosen_idxs['attention_mask']
        # negtive
        negtive = prompt_idxs['input_ids'] + rejected_idxs['input_ids']
        att_mask_neg = prompt_idxs['attention_mask'] + rejected_idxs['attention_mask']

        pos_actual_len = len(positive)
        neg_actual_len = len(negtive)
        # pad and truck
        positive = positive + [tokenizer.pad_token_id] * max(0, max_length - len(positive))
        att_mask_pos = att_mask_pos + [0] * max(0, max_length - len(att_mask_pos))
        negtive = negtive + [tokenizer.pad_token_id] * max(0, max_length - len(negtive))
        att_mask_neg = att_mask_neg + [0] * max(0, max_length - len(att_mask_neg))
        positive = positive[:max_length]
        att_mask_pos = att_mask_pos[:max_length]
        negtive = negtive[:max_length]
        att_mask_neg = att_mask_neg[:max_length]

        example['pos_actual_len'] = pos_actual_len
        example['neg_actual_len'] = neg_actual_len

        example['positive'] = positive
        example['negtive'] = negtive

        # example['prompt_idxs'] = prompt_idxs
        example['att_mask_pos'] = att_mask_pos
        example['att_mask_neg'] = att_mask_neg
        # example['chosen_idxs'] = chosen_idxs
        # example['rejected_idxs'] = rejected_idxs

        return example

    return tokenize

class data_collator_self(DataCollatorMixin):
    return_tensors: str = "pt"

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

        if labels is None:
            return batch

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        # print(batch)
        return batch