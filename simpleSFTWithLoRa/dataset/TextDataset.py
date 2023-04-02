import torch



class TextDataset(torch.utils.data.Dataset):

    def __init__(self,data):
        super(TextDataset,self).__init__()

        self.data = data


    def __getitem__(self, idx):

        item = self.data[idx]

        input_ids = item['input_ids']
        att_mask = item['attention_mask']
        labels = item['labels'] if 'labels' in item else None

        item = {
            'input_ids':torch.LongTensor(input_ids),
           "att_mask":torch.LongTensor(att_mask),
            "labels":torch.LongTensor(labels)
        }
        return item