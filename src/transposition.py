"""
Szyfr kolumnowy (columnar transposition).

Klucz to permutacja liczb 0..k-1 określająca kolejność odczytu kolumn.
Plaintext wpisujemy wierszami do macierzy o k kolumnach, w razie potrzeby
dopełniamy literą indeksu 0 ('A'), a ciphertext powstaje przez odczyt
kolumn w kolejności podanej kluczem.

Deszyfrowanie odbywa się przez permutację odwrotną (argsort).
"""

from __future__ import annotations
import numpy as np

PAD_LETTER = 0


def generate_key(key_length: int) -> np.ndarray:
    return np.random.permutation(key_length).astype(np.int8)


def _pad_to_multiple(text: np.ndarray, k: int) -> np.ndarray:
    rem = (-len(text)) % k
    if rem == 0:
        return text
    return np.concatenate([text, np.full(rem, PAD_LETTER, dtype=text.dtype)])


def encrypt(plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
    k = len(key)
    padded = _pad_to_multiple(plaintext, k)
    matrix = padded.reshape(-1, k)
    return matrix[:, key].T.reshape(-1)


def decrypt(ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
    k = len(key)
    if len(ciphertext) % k != 0:
        raise ValueError(
            f"Długość ciphertextu ({len(ciphertext)}) nie dzieli się przez "
            f"len(key)={k}."
        )
    rows = len(ciphertext) // k
    inv = np.argsort(key)
    matrix_reordered = ciphertext.reshape(k, rows).T
    return matrix_reordered[:, inv].reshape(-1)


def key_accuracy(true_key: np.ndarray, found_key: np.ndarray) -> float:
    return float(np.mean(true_key == found_key))


def text_accuracy(original: np.ndarray, recovered: np.ndarray) -> float:
    n = min(len(original), len(recovered))
    return float(np.mean(original[:n] == recovered[:n]))
