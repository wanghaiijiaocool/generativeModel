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


        positive = kwargs['positive']
        att_mask_pos = kwargs['att_mask_pos']
        pos = self.base_model(input_ids=positive,attention_mask = att_mask_pos)
        # use [CLS] or whatever the first token to classify
        logits_pos =torch.tanh(pos.logits[...,0])

        loss = None
        if("negtive" in kwargs):
            att_mask_neg = kwargs['att_mask_neg']
            negtive = kwargs['negtive']
            neg = self.base_model(input_ids=negtive, attention_mask=att_mask_neg)
            logits_neg = torch.tanh(neg.logits[..., 0])
            loss =  max(logits_neg - logits_pos, 0 )


        return transformers.utils.ModelOutput(
            loss=loss,
            score=logits_pos
        )

