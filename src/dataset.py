import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, target):
        self.texts = text
        self.targets = target

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        return {
            "text": text,
            "target": torch.tensor(target, dtype=torch.float).unsqueeze(0),
        }
