"""
Szyfrowanie i deszyfrowanie szyfrem podstawieniowym (monoalfabetycznym).

Klucz szyfrowania to bijekcja (permutacja) 26 liter alfabetu angielskiego.
Reprezentujemy klucze jako tablice numpy int8 o długości 26:
  encrypt_key[i] = j  oznacza: litera i (plaintext) -> litera j (ciphertext)
  decrypt_key[j] = i  oznacza: litera j (ciphertext) -> litera i (plaintext)

Referencja: Chen & Rosenthal (2010) "Decrypting Classical Cipher Text Using Markov Chain Monte Carlo"
"""

import string
import numpy as np

ALPHABET = string.ascii_uppercase
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def generate_key() -> np.ndarray:
    """Generuje losowy klucz szyfrowania (permutację 0-25)."""
    return np.random.permutation(26).astype(np.int8)


def inverse_key(key: np.ndarray) -> np.ndarray:
    """Oblicza odwrotną permutację (klucz deszyfrowania z klucza szyfrowania)."""
    inv = np.empty(26, dtype=np.int8)
    inv[key] = np.arange(26, dtype=np.int8)
    return inv


def encrypt(plaintext: np.ndarray, encrypt_key: np.ndarray) -> np.ndarray:
    """Szyfruje tablicę indeksów liter kluczem szyfrowania."""
    return encrypt_key[plaintext]


def decrypt(ciphertext: np.ndarray, decrypt_key: np.ndarray) -> np.ndarray:
    """Deszyfruje tablicę indeksów liter kluczem deszyfrowania."""
    return decrypt_key[ciphertext]


def key_accuracy(true_decrypt_key: np.ndarray, found_decrypt_key: np.ndarray) -> float:
    """Zwraca odsetek (0.0–1.0) pozycji, gdzie klucze deszyfrowania się zgadzają."""
    return float(np.mean(true_decrypt_key == found_decrypt_key))


def indices_to_str(indices: np.ndarray) -> str:
    """Konwertuje tablicę indeksów 0-25 na ciąg wielkich liter."""
    return "".join(ALPHABET[i] for i in indices)
