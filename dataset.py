from torch.utils.data import Dataset
from typing import Any, List, Tuple
from sklearn.model_selection import train_test_split


class JParaCrawlDataset(Dataset):
    def __init__(self, data_path: str, split: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.src, self.tgt = self.load_text()
        self.src_train, self.tgt_train, self.src_val, self.tgt_val, self.src_test, self.tgt_test = self.split_dataset()

    def load_text(self):
        tgt = list()
        src = list()
        with open(self.data_path, 'r') as f:
            cnt = 0
            while cnt <= 30000:
                txt_list = f.readline().split("\t")
                cnt += 1
                src.append(txt_list[4])
                tgt.append(txt_list[3])
        return src, tgt

    def split_dataset(self):
        src_train, src_test, tgt_train, tgt_test = train_test_split(
            self.src, self.tgt, test_size=0.1, random_state=42)
        src_train, src_val, tgt_train, tgt_val = train_test_split(
            src_train, tgt_train, test_size=0.1, random_state=42)

        return src_train, tgt_train, src_val, tgt_val, src_test, tgt_test

    def __getitem__(self, index: int) -> Tuple[List[str], List[str]]:
        if self.split == "train":
            return self.src_train[index], self.tgt_train[index]
        elif self.split == "valid":
            return self.src_val[index], self.tgt_val[index]
        else:
            return self.src_test[index], self.tgt_test[index]

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.src_train)
        elif self.split == "valid":
            return len(self.src_val)
        else:
            return len(self.src_test)


if __name__ == "__main__":
    train_datasets = JParaCrawlDataset(
        "datasets/en-ja/en-ja.bicleaner05.txt", split="train")
