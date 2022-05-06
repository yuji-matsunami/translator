from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from model import create_mask

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    
    src_batch = pad_sequence(src_batch, padding_value=1)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1)

    return src_batch, tgt_batch

def train_epoch(model, optimizer, loss_fn, src_ln, tgt_ln, batch_size, device):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(src_ln, tgt_ln))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_dataloader)

def evaluate(model, loss_fn, src_ln, tgt_ln, batch_size, device):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='val', language_pair=(src_ln, tgt_ln))
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)