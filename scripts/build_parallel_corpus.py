from pathlib import Path
import csv
import random


# ----- paths -----

# This file lives in scripts/, so parent[1] is the project root
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "parallel" / "raw_corpora"
OUT_DIR = ROOT / "data" / "parallel"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- loaders for each corpus -----


def load_manythings():
    """
    Load ManyThings/Tatoeba Japanese-English data from jpn.txt.

    Line format (tab separated), e.g.:
        Go.    行こう。    CC-BY 2.0 (France) Attribution: ...

    We take:
        English = column 0
        Japanese = column 1
    and return (jp, en) pairs.
    """
    path = RAW_DIR / "jpn-eng" / "jpn.txt"
    pairs: list[tuple[str, str]] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            en = parts[0].strip()
            jp = parts[1].strip()
            if not jp or not en:
                continue

            pairs.append((jp, en))

    print(f"Loaded {len(pairs):,} pairs from ManyThings (jpn.txt)")
    return pairs


def load_jesc():
    """
    Load OPUS JESC from the 'en-ja.txt' folder.

    Files:
      JESC.en-ja.ja -> Japanese
      JESC.en-ja.en -> English

    We zip the files line-by-line into (jp, en) pairs.
    """
    ja_path = RAW_DIR / "en-ja.txt" / "JESC.en-ja.ja"
    en_path = RAW_DIR / "en-ja.txt" / "JESC.en-ja.en"

    pairs: list[tuple[str, str]] = []

    with ja_path.open(encoding="utf-8") as ja_f, en_path.open(encoding="utf-8") as en_f:
        for jp, en in zip(ja_f, en_f):
            jp = jp.strip()
            en = en.strip()
            if not jp or not en:
                continue
            pairs.append((jp, en))

    print(f"Loaded {len(pairs):,} pairs from JESC (OPUS)")
    return pairs


# ----- cleaning + splitting -----


def clean_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Basic cleaning:
      - drop very short / very long lines
      - deduplicate (jp, en) pairs
    """
    max_len = 120  # characters
    min_len = 1

    def ok(s: str) -> bool:
        n = len(s)
        return min_len <= n <= max_len

    cleaned: list[tuple[str, str]] = []
    seen = set()

    for jp, en in pairs:
        if not (ok(jp) and ok(en)):
            continue

        key = jp + "\t" + en
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((jp, en))

    print(f"After cleaning & dedup: {len(cleaned):,} pairs")
    return cleaned


def split_and_save(pairs: list[tuple[str, str]]) -> None:
    """
    Shuffle and split into:
      - 90% train
      - 5% dev
      - 5% test
    and save as TSV files under data/parallel/.
    """
    random.seed(42)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)

    splits = {
        "train": pairs[:n_train],
        "dev": pairs[n_train:n_train + n_dev],
        "test": pairs[n_train + n_dev:],
    }

    for name, subset in splits.items():
        out_path = OUT_DIR / f"jp_en_{name}.tsv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for jp, en in subset:
                writer.writerow([jp, en])
        print(f"Wrote {len(subset):,} pairs to {out_path}")


# ----- main entry point -----


def main():
    many = load_manythings()
    jesc = load_jesc()

    all_pairs = many + jesc
    print(f"Total combined pairs (raw): {len(all_pairs):,}")

    cleaned = clean_pairs(all_pairs)
    split_and_save(cleaned)


if __name__ == "__main__":
    main()
