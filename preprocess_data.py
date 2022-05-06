from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List, Dict, Any, Tuple

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

def make_vocab_transform(unk_idx:int, src_ln:str, tgt_ln:str) -> Tuple[Dict, Dict]:
    """_summary_
        token_transformを作成
    Args:
        src_ln (str): _description_　ソースの言語
        tgt_rn (str): _description_　ターゲットの言語

    Returns:
        Dict[str, Any]: _description_ {"de": Vocab(), "en": Vocab()}
    """
    token_transform = {}
    vocab_transform = {}
    token_transform[src_ln] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[tgt_ln] = get_tokenizer('spacy', language='en_core_web_sm')

    special_simbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [src_ln, tgt_ln]:
        train_iter = Multi30k(split='train', language_pair=(src_ln, tgt_ln))
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_simbols,
                                                        special_first=True)

    for ln in [src_ln, tgt_ln]:
        vocab_transform[ln].set_default_index(unk_idx)
    return vocab_transform, token_transform

def main():
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_simbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_simbols,
                                                        special_first=True)

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)
    print(vocab_transform)



if __name__=="__main__":
    main()