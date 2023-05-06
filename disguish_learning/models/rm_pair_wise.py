"""
    build a pair-wised reward model to
    see if it is better than a rank based
    reward model.
"""

import transformers
from transformers import  AutoModel
import torch

class rm_pair(torch.nn.Module):
    def __init__(self,base_model:torch.nn.Module):
        super(rm_pair,self).__init__()
        # if(isinstance(base_model_name_or_path_or_base,str)):
        #     self.base_model =  AutoModel.from_pretrained(base_model_name_or_path_or_base,cache_dir=cache_dir)
        # else:
        #     raise Exception("not implemented")
        self.base_model = base_model
    def forward(self,**kwargs):
        input_ids = kwargs['input_ids']
        att_mask = kwargs['attention_mask']
        labels = kwargs['labels'] if 'labels' in kwargs else None

        output = self.base_model(input_ids=input_ids,attention_mask = att_mask)

        # use [CLS] or whatever the first token to classify
        logits = output.logits[...,0]

        loss = None
        if(labels is not None):
            loss_fct = torch.nn.BCELoss(size_average=True)
            loss = loss_fct(logits.view(-1).float(),labels.view(-1).float())

        return transformers.utils.ModelOutput(
            loss=loss,
            logits=logits
        )

