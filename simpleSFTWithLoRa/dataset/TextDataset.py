import torch



class TextDataset(torch.utils.data.Dataset):

    def __int__(self,data):
        super().__init__(TextDataset,self)

        self.data = data


    def __getitem__(self, idx):

        item = self.data[idx]

        input_ids = item['input_ids']
        att_mask = item['attention_mask']
        labels = item['labels'] if 'labels' in item else None

        item = {
            'input_ids':input_ids,
           "att_mask":att_mask,
            "labels":labels
        }
        return item