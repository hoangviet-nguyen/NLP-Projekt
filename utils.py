from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import os
import tarfile
import shutil
import torchaudio

class CommonVoiceDS(Dataset):

    def __init__(self, df_data: pd.DataFrame, root: str, transform=None, target_transform=None):
        self.df_data = df_data.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.df_data.shape[0]
    
    def __getitem__(self, index):
        sample = self.df_data.iloc[index]
        sentence = sample["sentence"]
        path = self.root + sample["path"]
        waveform, rate = torchaudio.load(path)

        if self.transform:
            waveform = self.transform(waveform)

        if self.target_transform:
            sentence = self.target_transform(sentence)

        return waveform, sentence
    


def download_data(data_path: str):

    download_path = hf_hub_download(
        repo_id="mozilla-foundation/common_voice_17_0",
        filename="audio/de/train/de_train_0.tar",
        repo_type="dataset",
        token=True
    )

    train_tsv = hf_hub_download(
        repo_id="mozilla-foundation/common_voice_17_0",
        filename="transcript/de/train.tsv",
        repo_type="dataset",
        token=True
    )

    print(f"Downloaded to: {download_path}")
    if not os.path.exists(data_path):
        with tarfile.open(download_path, "r") as tar:
            tar.extractall(path=data_path)
        shutil.copy(train_tsv, os.path.join(data_path, "train.tsv"))

    print(f"Extracted to: {data_path}")

    df_tsv = pd.read_table("./common_voice/train.tsv")
    filenames = os.listdir(data_path + "/de_train_0")
    df_train_0 = df_tsv[df_tsv["path"].isin(filenames)]
    df_train_0.to_csv("common_voice/train_0.tsv", sep="\t", index=False)
    print(f"Original: {df_tsv.shape[0]}, Set 0: {df_train_0.shape[0]}", "\n")
    print(df_tsv.columns)