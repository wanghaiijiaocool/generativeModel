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

        self.keys_to_ignore_on_save = []
        if(hasattr(self.base_model,'keys_to_ignore_on_save')):
            self.keys_to_ignore_on_save += list(getattr(self.base_model,self.base_model.keys_to_ignore_on_save))

        self._init_weights()
    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.scorer.weight,mean = 0., std = 0.01, a = -1., b = 1.)

    def pool_score(self,input_ids,hidden_states):
        batch_size = hidden_states.shape[0]
        sequence_lengths_pos = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(hidden_states.device)
        pool_logits_pos = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths_pos]
        logit = self.scorer(pool_logits_pos)
        #logits_pos = torch.tanh(logit)
        #print(pool_logits_pos)
        #print(logit)


        return logits_pos

    #@torchsnooper.snoop()
    def forward(self,**kwargs):


        positive = kwargs['positive']
        att_mask_pos = kwargs['att_mask_pos']

        pos = self.base_model(input_ids=positive,attention_mask = att_mask_pos)
        # use [CLS] or whatever the first token to classify
        logits_pos = self.pool_score(positive,pos.last_hidden_state)



        loss = None
        if("negtive" in kwargs):

            att_mask_neg = kwargs['att_mask_neg']
            negtive = kwargs['negtive']
            neg = self.base_model(input_ids=negtive, attention_mask=att_mask_neg)
            logits_neg = self.pool_score(negtive,neg.last_hidden_state)

            loss =  torch.sum(logits_neg - logits_pos)#torch.sum(torch.where(logits_neg - logits_pos > 0, logits_neg - logits_pos, 0 ))

            print(loss,logits_neg,logits_pos)
            print()

        return transformers.utils.ModelOutput(
            loss=loss,
            score=logits_pos
        )

