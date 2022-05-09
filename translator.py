from pyparsing import nums
import torch
from typing import Dict
from model import generate_square_subsequent_mask
from torchtext.data.metrics import bleu_score
from torchtext.datasets import Multi30k


class Translator:
    """
    #
    # 翻訳器クラス
    # 実際に翻訳したり
    # Bleuスコア計算したりするやつ
    # 
    """

    def __init__(self, model: torch.nn.Module, token_transform: Dict ,text_transform: Dict, vocab_transform: Dict, src_ln: str, tgt_ln: str, device: str, bos: int, eos: int) -> None:
        self.model = model
        self.token_transform = token_transform
        self.text_transform = text_transform
        self.vocab_transform = vocab_transform
        self.src_language = src_ln
        self.tgt_language = tgt_ln
        self.device = device
        self.BOS_IDX = bos
        self.EOS_IDX = eos

    def __greedy_decode(self, src, src_mask, max_len):
        start_symbol = self.BOS_IDX
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(
            torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys

    def translate(self, src_sentence):
        """翻訳を行う"""
        self.model.eval()
        src: torch.Tensor = self.text_transform[self.src_language](
            src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.__greedy_decode(
            src, src_mask, max_len=num_tokens + 5).flatten()
        return " ".join(self.vocab_transform[self.tgt_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    def get_bleu_score(self):
        """BLEU値を計算する"""
        self.model.eval()
        test_iter = Multi30k(split='test', language_pair=(
            self.src_language, self.tgt_language))
        candidate_corpus = list()  # 翻訳結果を入れる
        reference_corpus = list()  # 答えを入れる
        for s, t in test_iter:
            src: torch.Tensor = self.text_transform[self.src_language](
                s).view(-1, 1)
            num_tokens = src.shape[0]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
            tgt_tokens = self.__greedy_decode(
                src, src_mask, max_len=num_tokens+5).flatten()
            candidate_corpus.append(self.vocab_transform[self.tgt_language].lookup_tokens(list(tgt_tokens.cpu().numpy())))
            reference_corpus.append(['<bos>'] + self.token_transform[self.tgt_language](t) + ['<eos>'])
        print(reference_corpus[0])
        print(candidate_corpus[0])
        return bleu_score(candidate_corpus, [reference_corpus])

