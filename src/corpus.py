"""
Pobieranie i przetwarzanie korpusu tekstu („Lalka" Bolesława Prusa, Wolne Lektury).
Budowa macierzy log-częstości bigramów 26×26.

Wybór korpusu: „Lalka" to obszerna powieść prozą (ok. 1 mln znaków w dwóch tomach),
więc częstości bigramów są dużo stabilniejsze i bardziej reprezentatywne dla języka
niż próbka wierszowana (np. „Pan Tadeusz" — 13-zgłoskowiec zaburza naturalny rytm bigramów).
"""

from __future__ import annotations
import os
import urllib.request
import string
import numpy as np

# Dwa tomy „Lalki" z Wolnych Lektur. Łączymy w jeden strumień tekstu,
# co daje mocniejsze statystyki bigramów.
CORPUS_SOURCES = [
    ("lalka_tom_pierwszy.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt"),
    ("lalka_tom_drugi.txt",    "https://wolnelektury.pl/media/book/txt/lalka-tom-drugi.txt"),
]

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_HERE, "data")

ALPHABET = string.ascii_uppercase
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

# Polskie znaki diakrytyczne → ASCII
_POLISH_MAP = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def download_corpus() -> list[str]:
    """Pobiera pliki korpusu z Wolnych Lektur, jeśli nie są jeszcze na dysku."""
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = []
    for filename, url in CORPUS_SOURCES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Pobieranie korpusu:\n  {url}")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  -> {path}")
            except Exception as e:
                raise RuntimeError(
                    f"Nie można pobrać korpusu ({url}): {e}\n"
                    f"Pobierz ręcznie i zapisz jako:\n  {path}"
                ) from e
        paths.append(path)
    return paths


def load_corpus(paths: list[str]) -> str:
    """Wczytuje pliki korpusu i skleja je w jeden ciąg tekstu."""
    parts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


def strip_diacritics(text: str) -> str:
    """Zamienia polskie znaki diakrytyczne na ich ASCII odpowiedniki (ą→a, ł→l, …)."""
    return text.translate(_POLISH_MAP)


def text_to_indices(text: str) -> np.ndarray:
    """
    Konwertuje tekst na tablicę indeksów liter 0-25.
    Zamienia polskie znaki diakrytyczne na ASCII, ignoruje resztę nie-liter.
    """
    text = strip_diacritics(text).upper()
    return np.array([LETTER_TO_IDX[c] for c in text if c in LETTER_TO_IDX], dtype=np.int8)


def build_bigram_matrix(letter_indices: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """
    Buduje macierz log-częstości bigramów 26×26 z tablicy indeksów liter.

    Każdy element [i, j] to log P(j | i) — logarytm prawdopodobieństwa
    wystąpienia litery j bezpośrednio po literze i.

    Args:
        letter_indices: tablica indeksów liter (0-25)
        smoothing:      dodawany do każdego licznika (Laplace smoothing, unika log(0))

    Returns:
        macierz np.ndarray kształtu (26, 26) z wartościami log-prawdopodobieństw
    """
    counts = np.full((26, 26), smoothing, dtype=np.float64)
    np.add.at(counts, (letter_indices[:-1], letter_indices[1:]), 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.log(counts / row_sums)


def prepare_bigram_matrix() -> tuple[np.ndarray, np.ndarray]:
    """
    Pobiera korpus (jeśli trzeba), przetwarza go i zwraca:
      - macierz log-bigramów (26×26)
      - pełny tekst jako tablica indeksów liter
    """
    corpus_files = download_corpus()
    text = load_corpus(corpus_files)
    letter_indices = text_to_indices(text)
    print(f"Korpus załadowany: {len(letter_indices):,} liter")
    log_bigrams = build_bigram_matrix(letter_indices)
    return log_bigrams, letter_indices
