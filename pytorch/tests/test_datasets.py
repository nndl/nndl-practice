"""用本地小压缩包复现中断、缺文件和坏输入；测试过程不联网。"""
import gzip
import importlib.util
import io
from pathlib import Path
import random
import sys
import tarfile

import pytest


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "datasets" / "download.py"
    spec = importlib.util.spec_from_file_location("dataset_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {
        "ROOT": tmp_path, "DATASET_ROOT": tmp_path, "CIFAR_DIR": tmp_path,
        "IMDB_DIR": tmp_path / "imdb", "LCQMC_DIR": tmp_path / "lcqmc",
    }.items():
        monkeypatch.setattr(module, name, value)

    def no_network(*args, **kwargs):
        pytest.fail("test attempted a network download")

    monkeypatch.setattr(module.urllib.request, "urlretrieve", no_network)
    return module


def write_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries.items():
            data = content.encode("utf8")
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def lcqmc_archive(root, splits=("train", "dev", "test")):
    write_archive(root / "lcqmc.tar.gz", {
        f"lcqmc/{split}.tsv": "text_a\ttext_b\tlabel\n问题甲\t问题乙\t1\n"
        for split in splits
    })


def test_lcqmc_repairs_partial_output(downloader, tmp_path):
    lcqmc_archive(tmp_path)
    downloader.LCQMC_DIR.mkdir()
    (downloader.LCQMC_DIR / "train.txt.gz").write_bytes(b"interrupted gzip")
    downloader.fetch_lcqmc()
    for split in ("train", "dev", "test"):
        with gzip.open(downloader.LCQMC_DIR / f"{split}.txt.gz", "rt", encoding="utf8") as stream:
            assert stream.read() == "问题甲\t问题乙\t1\n"
    # 完整数据再次运行时不需要压缩包或网络。
    downloader.fetch_lcqmc()


def test_lcqmc_missing_split_does_not_publish_partial_dataset(downloader, tmp_path):
    lcqmc_archive(tmp_path, splits=("train", "dev"))
    with pytest.raises((ValueError, RuntimeError), match="test"):
        downloader.fetch_lcqmc()
    assert not (downloader.LCQMC_DIR / "train.txt.gz").exists()
    assert (tmp_path / "lcqmc.tar.gz").exists()


def test_lcqmc_malformed_row_does_not_replace_existing_data(downloader, tmp_path):
    write_archive(tmp_path / "lcqmc.tar.gz", {
        f"lcqmc/{split}.tsv": "甲\t乙\t1\n" if split != "test" else "甲\t乙\twrong\n"
        for split in ("train", "dev", "test")
    })
    downloader.LCQMC_DIR.mkdir()
    original = downloader.LCQMC_DIR / "train.txt.gz"
    original.write_bytes(b"original")
    with pytest.raises(ValueError, match="test"):
        downloader.fetch_lcqmc()
    assert original.read_bytes() == b"original"


def test_cifar_repairs_missing_batches_and_ignores_unrelated_paths(downloader, tmp_path):
    expected = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    entries = {f"cifar-10-batches-py/{name}": name for name in expected}
    entries["../outside.txt"] = "should not be extracted"
    write_archive(tmp_path / "cifar-10-python.tar.gz", entries)
    target = tmp_path / "cifar-10-batches-py"
    target.mkdir()
    (target / "batches.meta").write_bytes(b"old")
    downloader.fetch_cifar10()
    for name in expected:
        assert (target / name).read_text() == name
    assert not (tmp_path.parent / "outside.txt").exists()
    downloader.fetch_cifar10()


def test_unknown_dataset_is_error_before_any_fetch(downloader, monkeypatch):
    calls = []
    monkeypatch.setitem(downloader.FETCHERS, "imdb", lambda: calls.append("imdb"))
    monkeypatch.setattr(sys, "argv", ["download.py", "--only=imdb,typo"])
    with pytest.raises(SystemExit) as error:
        downloader.main()
    assert error.value.code != 0
    assert calls == []


def test_default_download_excludes_unused_boston(downloader, monkeypatch):
    calls = []
    for name in downloader.FETCHERS:
        monkeypatch.setitem(downloader.FETCHERS, name, lambda name=name: calls.append(name))
    monkeypatch.setattr(sys, "argv", ["download.py"])
    downloader.main()
    assert set(calls) == {"cifar10", "imdb", "lcqmc", "bert_vocab"}


def test_download_failure_cleans_partial_file(downloader, tmp_path, monkeypatch):
    def interrupted(url, dest, **kwargs):
        Path(dest).write_bytes(b"partial")
        raise OSError("connection interrupted")

    monkeypatch.setattr(downloader.urllib.request, "urlretrieve", interrupted)
    dest = tmp_path / "dataset.tar.gz"
    with pytest.raises(OSError, match="interrupted"):
        downloader._download("https://example.invalid/data", dest)
    assert not dest.exists()
    assert not dest.with_suffix(".gz.part").exists()


def test_imdb_preserves_split_and_training_only_vocab(downloader, tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "IMDB_SPLIT_SIZE", 8)
    monkeypatch.setattr(downloader, "IMDB_DEV_SIZE", 2)
    entries = {
        f"aclImdb/{split}/{label}/{i}.txt": f"  {split}{label}{i}<br />MOVIE\n"
        for split in ("train", "test") for label in ("pos", "neg") for i in range(4)
    }
    # 包内顺序不能影响训练/验证划分；unsup和目录外文件不进入实验。
    entries["aclImdb/train/unsup/extra.txt"] = "unsupervised"
    entries["../outside.txt"] = "outside"
    write_archive(tmp_path / "aclImdb_v1.tar.gz", dict(reversed(list(entries.items()))))
    downloader.IMDB_DIR.mkdir()
    with gzip.open(downloader.IMDB_DIR / "train.txt.gz", "wt") as stream:
        stream.write("old partial dataset")
    downloader.fetch_imdb()

    expected = [(str(label), f"train{group}{i} movie")
                for label, group in ((1, "pos"), (0, "neg")) for i in range(4)]
    random.Random(42).shuffle(expected)

    def read(name):
        with gzip.open(downloader.IMDB_DIR / f"{name}.txt.gz", "rt", encoding="utf8") as stream:
            return stream.read().splitlines()

    assert read("train") == [f"{label}\t{text}" for label, text in expected[2:]]
    assert read("dev") == [f"{label}\t{text}" for label, text in expected[:2]]
    assert len(read("test")) == 8
    vocab = read("vocab")
    assert vocab[:2] == ["[PAD]", "[UNK]"]
    assert set(vocab[2:]) == {word for _, text in expected[2:] for word in text.split()}
    assert not (tmp_path.parent / "outside.txt").exists()
    downloader.fetch_imdb()


def test_incomplete_imdb_archive_fails_before_publishing(downloader, tmp_path):
    write_archive(tmp_path / "aclImdb_v1.tar.gz", {"aclImdb/train/pos/0.txt": "good"})
    with pytest.raises(ValueError, match="expected"):
        downloader.fetch_imdb()
    assert not downloader.IMDB_DIR.exists()
    assert (tmp_path / "aclImdb_v1.tar.gz").exists()


def test_corrupt_gzip_tail_is_not_ready(downloader, tmp_path):
    path = tmp_path / "train.txt.gz"
    path.write_bytes(gzip.compress(b"complete-looking contents")[:-4])
    assert not downloader._gzip_ready(tmp_path, ["train"])


def test_bert_invalid_response_preserves_existing_file(downloader, tmp_path, monkeypatch):
    dest = tmp_path / "bert-base-chinese" / "vocab.txt"
    dest.parent.mkdir()
    dest.write_text("old invalid vocabulary", encoding="utf8")

    def bad_response(url, path, **kwargs):
        Path(path).write_text("<html>download error</html>", encoding="utf8")

    monkeypatch.setattr(downloader.urllib.request, "urlretrieve", bad_response)
    with pytest.raises(ValueError, match="vocabulary"):
        downloader.fetch_bert_vocab()
    assert dest.read_text(encoding="utf8") == "old invalid vocabulary"


def test_bert_replaces_invalid_cache_then_skips_valid_cache(downloader, tmp_path, monkeypatch):
    tokens = [f"token{i}" for i in range(21128)]
    tokens[0] = "[PAD]"
    tokens[100:104] = ["[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokens[343] = "\u2028"
    tokens[13502] = "##\u2028"
    calls = []

    def download_vocab(url, path, **kwargs):
        calls.append(url)
        Path(path).write_text("\n".join(tokens), encoding="utf8")

    monkeypatch.setattr(downloader.urllib.request, "urlretrieve", download_vocab)
    downloader.fetch_bert_vocab()
    downloader.fetch_bert_vocab()
    assert len(calls) == 1
    with (tmp_path / "bert-base-chinese" / "vocab.txt").open(encoding="utf8") as stream:
        assert [line.rstrip("\n") for line in stream] == tokens
