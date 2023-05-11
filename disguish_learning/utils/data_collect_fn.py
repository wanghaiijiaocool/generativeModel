#encoding=utf-8

def data_collator_self():
    def torch_call(self, features):
        import torch
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]



        batch = no_labels_features
        if labels is None:
            return batch

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch