"""
Szyfr kolumnowy (columnar transposition cipher).

Klucz = permutacja liczb 0..k-1 definiująca kolejność odczytu kolumn.

Procedura szyfrowania:
  1. Plaintext wpisujemy wierszami do macierzy o k kolumnach.
  2. W razie potrzeby dopełniamy literą 'A' (indeks 0) do pełnej liczby wierszy.
  3. Odczytujemy kolumny w kolejności zadanej kluczem i łączymy w ciphertext.

Przykład dla klucza [1, 2, 0], plaintext = "HELLOWORLD" (10 liter, k=3):
  - dopełnienie: "HELLOWORLD" + "AA"  -> "HELLOWORLDAA"  (12 = 4 × 3)
  - macierz po wpisaniu wierszami:
        H E L
        L O W
        O R L
        D A A
  - odczyt kolumn w kolejności [1, 2, 0]:
        kol 1 = E O R A
        kol 2 = L W L A
        kol 0 = H L O D
  - ciphertext = "EORALWLAHLOD"

Deszyfrowanie: odwrotna permutacja (argsort) odtwarza pierwotną kolejność kolumn.
"""

from __future__ import annotations
import numpy as np

# Dopełnienie tekstu do wielokrotności długości klucza (KISS: po prostu 'A').
# Padding wchodzi do bigramów jako szum o niewielkim udziale (≤ k-1 liter).
PAD_LETTER = 0


def generate_key(key_length: int) -> np.ndarray:
    """Losowa permutacja 0..key_length-1."""
    return np.random.permutation(key_length).astype(np.int8)


def _pad_to_multiple(text: np.ndarray, k: int) -> np.ndarray:
    """Dopełnia tablicę do wielokrotności k literą PAD_LETTER."""
    rem = (-len(text)) % k
    if rem == 0:
        return text
    return np.concatenate([text, np.full(rem, PAD_LETTER, dtype=text.dtype)])


def encrypt(plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Szyfruje tablicę indeksów liter szyfrem kolumnowym.
    Tekst jest dopełniany do wielokrotności len(key).
    """
    k = len(key)
    padded = _pad_to_multiple(plaintext, k)
    matrix = padded.reshape(-1, k)              # (rows, k) — wpisane wierszami
    return matrix[:, key].T.reshape(-1)         # kolumny w kolejności klucza


def decrypt(ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Deszyfruje szyfr kolumnowy. Wymaga len(ciphertext) % len(key) == 0.
    Zwraca tablicę tej samej długości co ciphertext (może zawierać padding).
    """
    k = len(key)
    if len(ciphertext) % k != 0:
        raise ValueError(
            f"Długość ciphertextu ({len(ciphertext)}) nie dzieli się przez "
            f"len(key)={k}."
        )
    rows = len(ciphertext) // k
    inv = np.argsort(key)
    matrix_reordered = ciphertext.reshape(k, rows).T   # (rows, k) == M[:, key]
    return matrix_reordered[:, inv].reshape(-1)


def key_accuracy(true_key: np.ndarray, found_key: np.ndarray) -> float:
    """Odsetek pozycji klucza, które się zgadzają (0.0–1.0)."""
    return float(np.mean(true_key == found_key))


def text_accuracy(original: np.ndarray, recovered: np.ndarray) -> float:
    """Odsetek liter zgodnych pozycyjnie między oryginałem a odszyfrowanym tekstem."""
    n = min(len(original), len(recovered))
    return float(np.mean(original[:n] == recovered[:n]))
