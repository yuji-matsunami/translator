import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List
from timeit import default_timer as timer
from preprocess_data import make_vocab_transform
from model import DEVICE, Seq2Seq
from train import train_epoch, evaluate, sarch_tensor_length
from translator import Translator


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))



if __name__ == "__main__":
    """
    # ハイパーパラメータの設定
    """
    SRC_LANGUAGE = 'ja'
    TGT_LANGUAGE = 'en'
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    torch.manual_seed(0)
    vocab_transform, token_transform = make_vocab_transform(
        UNK_IDX, src_ln=SRC_LANGUAGE, tgt_ln=TGT_LANGUAGE)
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 16
    NUM_ENCODE_LAYERS = 3
    NUM_DECODE_LAYERS = 3

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer = Seq2Seq(NUM_ENCODE_LAYERS, NUM_DECODE_LAYERS, EMB_SIZE, NHEAD,
                          SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # ハイパーパラメータを設定
    # seed固定しているのでいつも同じになる
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # tokenizeする
                                                   vocab_transform[ln], # 単語IDにする
                                                   tensor_transform) # BOS,EOSを追加する
    # model_path = "transformer.pth"
    # transformer.load_state_dict(torch.load(model_path))

    NUM_EPOCHS = 18
    # sarch_tensor_length(SRC_LANGUAGE, TGT_LANGUAGE, BATCH_SIZE, text_transform)
    """train"""
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn, SRC_LANGUAGE, TGT_LANGUAGE, BATCH_SIZE, DEVICE, text_transform)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn, SRC_LANGUAGE, TGT_LANGUAGE, BATCH_SIZE, DEVICE, text_transform)
        print(f"Epoch:{epoch}, Train loss:{train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

    # modelの保存
    model_path = "jp_to_en.pth"
    torch.save(transformer.state_dict(), model_path)
    exit()
    translator = Translator(transformer, token_transform, text_transform, vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, DEVICE, BOS_IDX, EOS_IDX)
    print("Eine Gruppe von Menschen steht vor einem Iglu .")
    print("↓↓↓↓↓↓↓↓↓↓↓")
    print(translator.translate(src_sentence="Eine Gruppe von Menschen steht vor einem Iglu ."))
    print(translator.get_bleu_score())