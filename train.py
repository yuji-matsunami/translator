from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from model import create_mask
from dataset import JParaCrawlDataset

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
    DATASET_PATH = "datasets/en-ja/en-ja.bicleaner05.txt"
    model.train()
    losses = 0
    # train_iter = Multi30k(split='train', language_pair=(src_ln, tgt_ln))
    train_iter = JParaCrawlDataset(DATASET_PATH, split='train')
    print(len(train_iter))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    cnt = 0
    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=1)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
        cnt += 1
    return losses / cnt

def evaluate(model, loss_fn, src_ln, tgt_ln, batch_size, device, txt_transform):
    global text_transform, SRC_LANGUAGE, TGT_LANGUAGE
    text_transform = txt_transform
    SRC_LANGUAGE = src_ln
    TGT_LANGUAGE = tgt_ln
    DATASET_PATH = "datasets/en-ja/en-ja.bicleaner05.txt"
    model.eval()
    losses = 0

    # val_iter = Multi30k(split='valid', language_pair=(src_ln, tgt_ln))
    val_iter = JParaCrawlDataset(DATASET_PATH, split='valid')
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)
    
    cnt = 0
    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=1)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        cnt += 1

    return losses / cnt