from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, data):  # 需要直接把数据给到MyDataset
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


