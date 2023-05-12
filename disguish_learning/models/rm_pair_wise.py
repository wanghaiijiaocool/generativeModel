"""
    build a pair-wised reward model to
    see if it is better than a rank based
    reward model.
"""

import transformers
from transformers import  AutoModel
import torch
import torchsnooper

class rm_pair(torch.nn.Module):
    def __init__(self,base_model:torch.nn.Module):
        super(rm_pair,self).__init__()
        # if(isinstance(base_model_name_or_path_or_base,str)):
        #     self.base_model =  AutoModel.from_pretrained(base_model_name_or_path_or_base,cache_dir=cache_dir)
        # else:
        #     raise Exception("not implemented")
        self.base_model = base_model
        self.config = base_model.config
        self.scorer = torch.nn.Linear(self.config.hidden_size,1,bias=False)
    #@torchsnooper.snoop()
    def forward(self,**kwargs):


        positive = kwargs['positive']
        att_mask_pos = kwargs['att_mask_pos']
        pos = self.base_model(input_ids=positive,attention_mask = att_mask_pos)
        # use [CLS] or whatever the first token to classify
        logits_pos =torch.tanh(self.scorer(pos.last_hidden_state[:,0,:]))

        loss = None
        if("negtive" in kwargs):
            att_mask_neg = kwargs['att_mask_neg']
            negtive = kwargs['negtive']
            neg = self.base_model(input_ids=negtive, attention_mask=att_mask_neg)
            logits_neg = torch.tanh(self.scorer(neg.last_hidden_state[:,0,:]))
            loss =  torch.sum(torch.where(logits_neg - logits_pos > 0, logits_neg - logits_pos, 0 ))


        return transformers.utils.ModelOutput(
            loss=loss,
            score=logits_pos
        )

