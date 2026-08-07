from __future__ import annotations
import os
import urllib.request
import string
import numpy as np

CORPUS_SOURCES = {
    "lalka": [
        ("lalka_tom_pierwszy.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt"),
        ("lalka_tom_drugi.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-drugi.txt"),
    ],
    "quo_vadis": [
        ("quo_vadis.txt", "https://wolnelektury.pl/media/book/txt/quo-vadis.txt"),
    ],
}

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_HERE, "data")

ALPHABET = string.ascii_uppercase
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

_POLISH_MAP = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def download_corpus(name: str = "lalka") -> list[str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    if name not in CORPUS_SOURCES:
        raise ValueError(f"Nieznany korpus: {name}. Dostępne: {list(CORPUS_SOURCES.keys())}")

    paths = []
    for filename, url in CORPUS_SOURCES[name]:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Pobieranie korpusu ({name}): {url}")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                raise RuntimeError(
                    f"Nie można pobrać korpusu ({url}): {e}. "
                    f"Pobierz ręcznie i zapisz jako: {path}"
                ) from e
        paths.append(path)
    return paths


def load_corpus(paths: list[str]) -> str:
    parts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


def strip_diacritics(text: str) -> str:
    return text.translate(_POLISH_MAP)


def text_to_indices(text: str) -> np.ndarray:
    text = strip_diacritics(text).upper()
    return np.array([LETTER_TO_IDX[c] for c in text if c in LETTER_TO_IDX], dtype=np.int8)


def build_bigram_matrix(letter_indices: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    counts = np.full((26, 26), smoothing, dtype=np.float64)
    np.add.at(counts, (letter_indices[:-1], letter_indices[1:]), 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.log(counts / row_sums)


def prepare_bigram_matrix(name: str = "lalka", test_split: float = 0.1):
    corpus_files = download_corpus(name)
    text = load_corpus(corpus_files)
    letter_indices = text_to_indices(text)

    split_idx = int(len(letter_indices) * (1 - test_split))
    train_text = letter_indices[:split_idx]
    test_text = letter_indices[split_idx:]

    print(f"Korpus '{name}' załadowany: {len(letter_indices):,} liter")
    print(f"  Trening: {len(train_text):,} liter, Test: {len(test_text):,} liter")

    log_bigrams = build_bigram_matrix(train_text)
    return log_bigrams, train_text, test_text
