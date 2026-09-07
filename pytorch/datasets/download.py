"""准备第5、6、8章的数据，默认下载CIFAR-10、IMDB、LCQMC和BERT字表。

在仓库根运行：python pytorch/datasets/download.py --only=imdb,lcqmc
数据写到本脚本所在目录。完整文件会跳过；缺失或损坏的gzip文件会重新准备。
第2章加州房价由sklearn获取；旧Boston接口仅保留供显式调用。
"""
import argparse
from collections import Counter
import gzip
import io
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory
import urllib.request
import zlib


DATASET_ROOT = Path(__file__).resolve().parent
ROOT = DATASET_ROOT.parents[1]
CIFAR_DIR = DATASET_ROOT
IMDB_DIR = DATASET_ROOT / "imdb"
LCQMC_DIR = DATASET_ROOT / "lcqmc"
IMDB_SPLIT_SIZE = 25000
IMDB_DEV_SIZE = 5000

SOURCES = {
    "boston": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "imdb": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "lcqmc": "https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz",
    "bert_vocab": "https://modelscope.cn/api/v1/models/tiansz/bert-base-chinese/repo?Revision=master&FilePath=vocab.txt",
}
DEFAULT_DATASETS = ("cifar10", "imdb", "lcqmc", "bert_vocab")


def log(msg):
    print(f"[download] {msg}", flush=True)


def _nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def _gzip_ready(folder, names):
    """检查全部必需文件，并读到gzip尾部验证CRC，识别上次中断的产物。"""
    for name in names:
        path = folder / f"{name}.txt.gz"
        if not _nonempty(path):
            return False
        try:
            with gzip.open(path, "rb") as stream:
                if not stream.read(1):
                    return False
                while stream.read(1024 * 1024):
                    pass
        except (OSError, EOFError, zlib.error):
            return False
    return True


def _download(url, dest):
    if _nonempty(dest):
        log(f"skip (exists): {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log(f"GET {url}\n  -> {dest.relative_to(ROOT)}")

    def hook(blocks, block_size, total_size):
        if total_size > 0:
            done = min(blocks * block_size, total_size)
            sys.stdout.write(f"\r  {done * 100 // total_size}% "
                             f"({done / 1024**2:.1f}/{total_size / 1024**2:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        if not _nonempty(tmp):
            raise ValueError(f"empty download: {url}")
        tmp.replace(dest)
        print()
    finally:
        tmp.unlink(missing_ok=True)


def _regular_members(archive):
    # 仅读普通文件，绝不按压缩包提供的路径落盘或跟随链接。
    return {str(PurePosixPath(member.name)): member
            for member in archive.getmembers() if member.isfile()}


def _publish(stage, target, names):
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        (stage / name).replace(target / name)


def fetch_boston():
    """旧实验兼容入口；当前第2章使用加州房价，默认不下载此文件。"""
    _download(SOURCES["boston"], DATASET_ROOT / "boston_house_prices.csv")


def fetch_cifar10():
    target = CIFAR_DIR / "cifar-10-batches-py"
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    if all(_nonempty(target / name) for name in names):
        log(f"skip (complete): {target.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "cifar-10-python.tar.gz"
    _download(SOURCES["cifar10"], tar_path)
    with TemporaryDirectory(prefix="_cifar_prepare_", dir=DATASET_ROOT) as temp:
        stage = Path(temp)
        with tarfile.open(tar_path, "r:gz") as archive:
            members = _regular_members(archive)
            for name in names:
                member = members.get(f"cifar-10-batches-py/{name}")
                if member is None or member.size == 0:
                    raise ValueError(f"CIFAR-10 archive missing nonempty {name}")
                with archive.extractfile(member) as src, (stage / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        _publish(stage, target, names)
    tar_path.unlink(missing_ok=True)
    log("CIFAR-10 ready")


def fetch_imdb():
    splits = ("train", "dev", "test", "vocab")
    if _gzip_ready(IMDB_DIR, splits):
        log(f"skip (complete): {IMDB_DIR.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "aclImdb_v1.tar.gz"
    _download(SOURCES["imdb"], tar_path)
    # 直接从压缩包读取影评，避免解压数万个小文件和误用未解压完的目录。
    reviews = {(split, label): {} for split in ("train", "test") for label in ("pos", "neg")}
    # gzip只能高效顺序解压。先依包内顺序读取，再按文件名排序以保持原来的划分。
    with tarfile.open(tar_path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 4 or parts[0] != "aclImdb"
                    or (parts[1], parts[2]) not in reviews or not parts[3].endswith(".txt")):
                continue
            group = reviews[parts[1], parts[2]]
            if parts[3] in group:
                raise ValueError(f"IMDB duplicate review: {member.name}")
            with archive.extractfile(member) as stream:
                text = stream.read().decode("utf-8")
            text = re.sub(r"\s+", " ", text.replace("<br />", " ")).strip().lower()
            if not text:
                raise ValueError(f"IMDB empty review: {member.name}")
            group[parts[3]] = text

    def collect(split):
        items = []
        for label, label_dir in (("1", "pos"), ("0", "neg")):
            group = reviews[split, label_dir]
            if len(group) != IMDB_SPLIT_SIZE // 2:
                raise ValueError(f"IMDB {split}/{label_dir}: expected "
                                 f"{IMDB_SPLIT_SIZE // 2} reviews, got {len(group)}")
            items.extend((label, group[name]) for name in sorted(group))
        return items

    train_all, test_items = collect("train"), collect("test")

    random.Random(42).shuffle(train_all)
    dev_items, train_items = train_all[:IMDB_DEV_SIZE], train_all[IMDB_DEV_SIZE:]
    counter = Counter(word for _, text in train_items for word in text.split())
    vocab_tokens = ["[PAD]", "[UNK]"] + [word for word, _ in counter.most_common(50000)]

    with TemporaryDirectory(prefix="_imdb_prepare_", dir=DATASET_ROOT) as temp:
        stage = Path(temp)
        for name, items in (("train", train_items), ("dev", dev_items), ("test", test_items)):
            with gzip.open(stage / f"{name}.txt.gz", "wt", encoding="utf-8") as stream:
                for label, text in items:
                    stream.write(f"{label}\t{text}\n")
        with gzip.open(stage / "vocab.txt.gz", "wt", encoding="utf-8") as stream:
            stream.write("\n".join(vocab_tokens) + "\n")
        _publish(stage, IMDB_DIR, [f"{name}.txt.gz" for name in splits])
    tar_path.unlink(missing_ok=True)
    log("IMDB ready")


def fetch_lcqmc():
    splits = ("train", "dev", "test")
    if _gzip_ready(LCQMC_DIR, splits):
        log(f"skip (complete): {LCQMC_DIR.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "lcqmc.tar.gz"
    _download(SOURCES["lcqmc"], tar_path)
    with TemporaryDirectory(prefix="_lcqmc_prepare_", dir=DATASET_ROOT) as temp:
        stage = Path(temp)
        with tarfile.open(tar_path, "r:gz") as archive:
            members = _regular_members(archive)
            parents = {PurePosixPath(name).parent for name in members
                       if PurePosixPath(name).name in ("train.tsv", "train.txt")}
            if len(parents) != 1:
                raise ValueError("LCQMC archive must contain exactly one train.tsv/train.txt directory")
            parent = parents.pop()
            for split in splits:
                member = next((members[str(parent / f"{split}.{ext}")]
                               for ext in ("tsv", "txt")
                               if str(parent / f"{split}.{ext}") in members), None)
                if member is None:
                    raise ValueError(f"LCQMC archive missing {split}.tsv/.txt")
                rows = 0
                with archive.extractfile(member) as raw, \
                        io.TextIOWrapper(raw, encoding="utf-8-sig") as src, \
                        gzip.open(stage / f"{split}.txt.gz", "wt", encoding="utf-8") as dst:
                    for line_number, line in enumerate(src, 1):
                        parts = line.rstrip("\r\n").split("\t")
                        if line_number == 1 and len(parts) == 3 and parts[-1].lower() == "label":
                            continue
                        if len(parts) != 3 or not parts[0].strip() or not parts[1].strip() or parts[2] not in ("0", "1"):
                            raise ValueError(f"LCQMC {split}:{line_number}: expected text_a, text_b, label (0/1)")
                        dst.write("\t".join(parts) + "\n")
                        rows += 1
                if rows == 0:
                    raise ValueError(f"LCQMC {split}: no examples")
        _publish(stage, LCQMC_DIR, [f"{split}.txt.gz" for split in splits])
    tar_path.unlink(missing_ok=True)
    log("LCQMC ready")


def _valid_bert_vocab(path):
    if not _nonempty(path):
        return False
    try:
        # 字表包含U+2028及其WordPiece形式；splitlines()会把词元误拆成多行。
        with path.open(encoding="utf-8") as stream:
            words = [line.rstrip("\n") for line in stream]
    except UnicodeError:
        return False
    return (len(words) == 21128 and len(set(words)) == len(words)
            and words[0] == "[PAD]" and words[100:104] == ["[UNK]", "[CLS]", "[SEP]", "[MASK]"])


def fetch_bert_vocab():
    """字符级实验使用固定字表；不包含模型权重或WordPiece分词器。"""
    dest = DATASET_ROOT / "bert-base-chinese" / "vocab.txt"
    if _valid_bert_vocab(dest):
        log(f"skip (complete): {dest.relative_to(ROOT)}")
        return
    with TemporaryDirectory(prefix="_bert_prepare_", dir=DATASET_ROOT) as temp:
        candidate = Path(temp) / "vocab.txt"
        _download(SOURCES["bert_vocab"], candidate)
        if not _valid_bert_vocab(candidate):
            raise ValueError("BERT vocabulary validation failed: expected 21128 unique tokens and standard special-token IDs")
        dest.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(dest)
    log("BERT Chinese vocabulary ready (21128 tokens)")


FETCHERS = {
    "boston": fetch_boston, "cifar10": fetch_cifar10, "imdb": fetch_imdb,
    "lcqmc": fetch_lcqmc, "bert_vocab": fetch_bert_vocab,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated subset: " + ",".join(FETCHERS))
    args = parser.parse_args()
    selected = list(dict.fromkeys(k.strip() for k in args.only.split(",") if k.strip())) or list(DEFAULT_DATASETS)
    unknown = [name for name in selected if name not in FETCHERS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; valid: {', '.join(FETCHERS)}")
    DATASET_ROOT.mkdir(exist_ok=True)
    for name in selected:
        log(f"=== {name} ===")
        FETCHERS[name]()
    log("all done")


if __name__ == "__main__":
    main()
