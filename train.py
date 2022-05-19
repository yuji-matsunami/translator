import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from model import create_mask
from dataset import JParaCrawlDataset, KFTTDataset
from tqdm import tqdm
import logging

text_transform = {}
SRC_LANGUAGE = ""
TGT_LANGUAGE = ""
def collate_fn(batch):
    global text_transform, SRC_LANGUAGE, TGT_LANGUAGE
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    
    src_batch = pad_sequence(src_batch, padding_value=1)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1)

    return src_batch, tgt_batch

def train_epoch(model, optimizer, loss_fn, src_ln, tgt_ln, batch_size, device, txt_transform):
    global text_transform, SRC_LANGUAGE, TGT_LANGUAGE
    text_transform = txt_transform
    SRC_LANGUAGE = src_ln
    TGT_LANGUAGE = tgt_ln
    DATASET_PATH = "datasets/kftt-data-1.0/data/orig"
    model.train()
    losses = 0
    # train_iter = Multi30k(split='train', language_pair=(src_ln, tgt_ln))
    # train_iter = JParaCrawlDataset(DATASET_PATH, split='train')
    train_iter = KFTTDataset(DATASET_PATH, split="train")
    logging.basicConfig(filename='logfile.log', level=logging.INFO)
    print(len(train_iter))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    cnt = 0
    for src, tgt in tqdm(train_dataloader):
        logging.info('src:%d, tgt:%d', src.size(0), tgt.size(0))
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=1)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        del src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
        del loss, tgt_out # 使わなくなった計算グラフを削除
        torch.cuda.empty_cache()
        cnt += 1
    return losses / cnt

def evaluate(model, loss_fn, src_ln, tgt_ln, batch_size, device, txt_transform):
    global text_transform, SRC_LANGUAGE, TGT_LANGUAGE
    text_transform = txt_transform
    SRC_LANGUAGE = src_ln
    TGT_LANGUAGE = tgt_ln
    DATASET_PATH = "datasets/kftt-data-1.0/data/orig"
    model.eval()
    losses = 0

    # val_iter = Multi30k(split='valid', language_pair=(src_ln, tgt_ln))
    # val_iter = JParaCrawlDataset(DATASET_PATH, split='valid')
    val_iter = KFTTDataset(DATASET_PATH, split="valid")
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)
    
    cnt = 0
    try:
        for src, tgt in tqdm(val_dataloader):
            if src.size(0) >= 400: continue

            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=1)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            del src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            del loss, tgt_out
            torch.cuda.empty_cache()
            cnt += 1
    except RuntimeError:
        print(src.size(0))
        print(tgt.size(0))
    return losses / cnt

def sarch_tensor_length(src_ln, tgt_ln, batch_size, txt_transform):
    global text_transform, SRC_LANGUAGE, TGT_LANGUAGE
    text_transform = txt_transform
    SRC_LANGUAGE = src_ln
    TGT_LANGUAGE = tgt_ln
    DATASET_PATH = "datasets/kftt-data-1.0/data/orig"
    train_iter = KFTTDataset(DATASET_PATH, split='train')
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    src_lengths = list()
    tgt_lengths = list()
    for src, tgt in tqdm(train_dataloader):
        src_lengths.append(src.size(0))
        tgt_lengths.append(tgt.size(0))
    print(sorted(src_lengths))
    print(sorted(tgt_lengths))
