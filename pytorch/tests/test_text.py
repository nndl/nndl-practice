"""文本输入需在训练与预测时保持同一约定。"""
import gzip

import torch

from nndl.text import load_imdb_data, load_vocab, tokenize
from nndl.rnn import MySRN


def test_tokenize_matches_download_preprocessing():
    assert tokenize("  GREAT<br />movie!\n AGAIN\t") == ["great", "movie!", "again"]
    assert tokenize(" \n\t") == []


def test_gzip_text_and_vocabulary_round_trip(tmp_path):
    for split in ("train", "dev", "test"):
        with gzip.open(tmp_path / f"{split}.txt.gz", "wt", encoding="utf8") as stream:
            stream.write("1\tgreat movie\n0\tbad movie\n")
    with gzip.open(tmp_path / "vocab.txt.gz", "wt", encoding="utf8") as stream:
        stream.write("[PAD]\n[UNK]\ngreat\nmovie\nbad\n")
    train, dev, test = load_imdb_data(tmp_path)
    assert train == dev == test == [("great movie", 1), ("bad movie", 0)]
    assert load_vocab(tmp_path / "vocab.txt.gz")["[UNK]"] == 1


def test_srn_state_follows_parameter_dtype():
    model = MySRN(embed=4, hidden=5).double()
    output = model(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    assert output.dtype == torch.double
    output.sum().backward()
    assert model.Wh.grad is not None
