"""下载本书各章节所需的数据集到本地。

每次运行会跳过已存在的文件（按目标路径判断），可以反复运行。

数据放在本脚本所在的 pytorch/datasets/ 下，各章 notebook 用相对路径 ../datasets/... 访问：
  pytorch/datasets/boston_house_prices.csv          — chap2
  pytorch/datasets/cifar-10-batches-py/             — chap5
  pytorch/datasets/imdb/{train,dev,test,vocab}.txt.gz  — chap6 / chap8（共享）
  pytorch/datasets/lcqmc/{train,dev,test}.txt.gz    — chap8
  pytorch/datasets/bert-base-chinese/vocab.txt      — chap8（LCQMC Transformer 精确复刻，可选）

MNIST 不在此处下载——`torchvision.datasets.MNIST(download=True)` 会自动下载。
Iris 通过 `sklearn.datasets.load_iris()` 直接取，无需下载。

用法（在仓库根目录 nndl-practice 下）：
  python pytorch/datasets/download.py
  python pytorch/datasets/download.py --only=lcqmc,bert_vocab   # 只下载指定数据集
"""
import argparse
import gzip
import os
import random
import re
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent      # nndl-practice/pytorch/datasets（脚本所在目录）
ROOT = DATASET_ROOT.parents[1]                       # nndl-practice（仅用于日志相对路径）
CIFAR_DIR = DATASET_ROOT  # cifar-10-batches-py/ 直接落在 datasets/ 下
IMDB_DIR = DATASET_ROOT / "imdb"
LCQMC_DIR = DATASET_ROOT / "lcqmc"

SOURCES = {
    "boston": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "imdb": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "lcqmc": "https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz",
    # BERT 中文 vocab.txt（21128）——chap8 LCQMC Transformer 精确复刻用；ModelScope 国内可达
    "bert_vocab": "https://modelscope.cn/api/v1/models/tiansz/bert-base-chinese/repo?Revision=master&FilePath=vocab.txt",
}


def log(msg):
    print(f"[download] {msg}", flush=True)


def _download(url: str, dest: Path):
    if dest.exists():
        log(f"skip (exists): {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"GET {url}")
    log(f"  -> {dest.relative_to(ROOT)}")

    def hook(blocks, block_size, total_size):
        if total_size > 0:
            done = blocks * block_size
            pct = min(100, done * 100 // total_size)
            mb_done = done / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)


# ----------------- Boston housing -----------------
def fetch_boston():
    dest = DATASET_ROOT / "boston_house_prices.csv"
    _download(SOURCES["boston"], dest)
    # The selva86 CSV uses lowercase column names; chap2 expects boston_house_prices.csv format.
    # Both work with pandas; no schema rewrite needed.


# ----------------- CIFAR-10 -----------------
def fetch_cifar10():
    target_dir = CIFAR_DIR / "cifar-10-batches-py"
    if target_dir.exists() and any(target_dir.iterdir()):
        log(f"skip (exists): {target_dir.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "cifar-10-python.tar.gz"
    _download(SOURCES["cifar10"], tar_path)
    log(f"extracting CIFAR-10 -> {CIFAR_DIR.relative_to(ROOT)}")
    CIFAR_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(CIFAR_DIR)
    # remove the tarball; pickled batches under cifar-10-batches-py/ are the consumed data
    tar_path.unlink(missing_ok=True)
    log("CIFAR-10 ready")


# ----------------- IMDB -----------------
def fetch_imdb():
    # Already processed?
    if (IMDB_DIR / "train.txt.gz").exists():
        log(f"skip (exists): {(IMDB_DIR / 'train.txt.gz').relative_to(ROOT)}")
        return

    tar_path = DATASET_ROOT / "aclImdb_v1.tar.gz"
    extract_dir = DATASET_ROOT / "_aclImdb_extract"
    _download(SOURCES["imdb"], tar_path)

    if not extract_dir.exists():
        log("extracting aclImdb (this takes a minute, ~50k small files)...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        log("aclImdb extracted")

    log("processing IMDB into train/dev/test.txt + vocab.txt")
    aclImdb = extract_dir / "aclImdb"

    def collect(split):
        items = []
        for label, label_dir in [("1", "pos"), ("0", "neg")]:
            d = aclImdb / split / label_dir
            for fp in sorted(d.glob("*.txt")):
                text = fp.read_text(encoding="utf-8", errors="ignore")
                # crude cleanup: <br /> -> space, drop control chars
                text = text.replace("<br />", " ")
                text = re.sub(r"\s+", " ", text).strip().lower()
                items.append((label, text))
        return items

    train_all = collect("train")  # 25k
    test_items = collect("test")  # 25k

    rng = random.Random(42)
    rng.shuffle(train_all)
    dev_size = 5000
    dev_items = train_all[:dev_size]
    train_items = train_all[dev_size:]

    def write_split(items, path):
        # gzip-compressed text file
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fw:
            for label, text in items:
                fw.write(f"{label}\t{text}\n")

    # Build vocab from train (top 50k words by frequency, plus special tokens)
    counter = Counter()
    for _, text in train_items:
        counter.update(text.split(" "))
    vocab_tokens = ["[PAD]", "[UNK]"] + [w for w, _ in counter.most_common(50000) if w]

    write_split(train_items, IMDB_DIR / "train.txt.gz")
    write_split(dev_items, IMDB_DIR / "dev.txt.gz")
    write_split(test_items, IMDB_DIR / "test.txt.gz")
    vocab_path = IMDB_DIR / "vocab.txt.gz"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(vocab_path, "wt", encoding="utf-8") as fw:
        for tok in vocab_tokens:
            fw.write(tok + "\n")
    log(f"  written: {IMDB_DIR.relative_to(ROOT)} (*.txt.gz)")

    # Clean up extracted folder and tarball to save disk
    log("cleaning up extracted aclImdb and aclImdb_v1.tar.gz...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    log("IMDB ready")


# ----------------- LCQMC -----------------
def fetch_lcqmc():
    target_dir = LCQMC_DIR
    train_path = target_dir / "train.txt.gz"
    if train_path.exists():
        log(f"skip (exists): {train_path.relative_to(ROOT)}")
        return
    tar_path = DATASET_ROOT / "lcqmc.tar.gz"
    extract_dir = DATASET_ROOT / "_lcqmc_extract"
    _download(SOURCES["lcqmc"], tar_path)

    log("extracting LCQMC")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)

    # paddle hub's tar typically contains lcqmc/{train,dev,test}.tsv with header
    # Find the actual layout and normalize to chap8 expected format.
    candidates = list(extract_dir.rglob("train.*"))
    log(f"  LCQMC extracted entries: {[p.relative_to(extract_dir) for p in candidates]}")
    src_dir = candidates[0].parent if candidates else None
    if src_dir is None:
        raise RuntimeError("LCQMC tar didn't produce a train.* file")

    target_dir.mkdir(parents=True, exist_ok=True)

    def copy_normalize(src: Path, dst: Path):
        # write gzip-compressed
        with src.open("r", encoding="utf-8") as fr, gzip.open(dst, "wt", encoding="utf-8") as fw:
            for i, line in enumerate(fr):
                # paddle hub format: text_a\ttext_b\tlabel  (with possible header on line 0)
                parts = line.rstrip("\n").split("\t")
                if i == 0 and not (parts and parts[-1].isdigit()):
                    continue  # header line
                fw.write(line if line.endswith("\n") else line + "\n")

    for split in ("train", "dev", "test"):
        src = src_dir / f"{split}.tsv"
        if not src.exists():
            src = src_dir / f"{split}.txt"
        if not src.exists():
            log(f"  warning: {split} file not found in {src_dir}")
            continue
        copy_normalize(src, target_dir / f"{split}.txt.gz")
        log(f"  written: {(target_dir / f'{split}.txt.gz').relative_to(ROOT)}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    log("LCQMC ready")


# ----------------- BERT 中文 vocab.txt（LCQMC Transformer 精确复刻，可选）-----------------
def fetch_bert_vocab():
    """下载 bert-base-chinese vocab.txt 供 chap8 LCQMC Transformer 精确复刻。
    非致命：缺它时 rerun_lcqmc_transformer.py 会自动退回字符级词表。"""
    dest = DATASET_ROOT / "bert-base-chinese" / "vocab.txt"
    if dest.exists():
        log(f"skip (exists): {dest.relative_to(ROOT)}")
        return
    try:
        _download(SOURCES["bert_vocab"], dest)
        lines = dest.read_text(encoding="utf-8").splitlines()
        if len(lines) < 21000 or "[CLS]" not in lines:
            raise RuntimeError(f"校验失败（lines={len(lines)}）")
        log(f"BERT 中文 vocab.txt ready ({len(lines)} 行)")
    except Exception as e:
        dest.unlink(missing_ok=True)
        log(f"  warning: bert_vocab 获取失败（{type(e).__name__}: {e}）—— "
            f"可手动放到 {dest.relative_to(ROOT)}；缺它 LCQMC 脚本退回字符级词表")


# ----------------- main -----------------
FETCHERS = {
    "boston": fetch_boston,
    "cifar10": fetch_cifar10,
    "imdb": fetch_imdb,
    "lcqmc": fetch_lcqmc,
    "bert_vocab": fetch_bert_vocab,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated subset, e.g. boston,lcqmc",
    )
    args = parser.parse_args()
    DATASET_ROOT.mkdir(exist_ok=True)
    selected = [k.strip() for k in args.only.split(",") if k.strip()] or list(FETCHERS.keys())
    for name in selected:
        if name not in FETCHERS:
            log(f"unknown dataset '{name}', skipping (valid: {list(FETCHERS.keys())})")
            continue
        log(f"=== {name} ===")
        try:
            FETCHERS[name]()
        except Exception as e:
            log(f"  ERROR for {name}: {type(e).__name__}: {e}")
            raise
    log("all done")


if __name__ == "__main__":
    main()
