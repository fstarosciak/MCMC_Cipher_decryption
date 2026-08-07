import string
import numpy as np

ALPHABET = string.ascii_uppercase
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def generate_key() -> np.ndarray:
    return np.random.permutation(26).astype(np.int8)


def inverse_key(key: np.ndarray) -> np.ndarray:
    inv = np.empty(26, dtype=np.int8)
    inv[key] = np.arange(26, dtype=np.int8)
    return inv


def encrypt(plaintext: np.ndarray, encrypt_key: np.ndarray) -> np.ndarray:
    return encrypt_key[plaintext]


def decrypt(ciphertext: np.ndarray, decrypt_key: np.ndarray) -> np.ndarray:
    return decrypt_key[ciphertext]


def key_accuracy(true_decrypt_key: np.ndarray, found_decrypt_key: np.ndarray) -> float:
    return float(np.mean(true_decrypt_key == found_decrypt_key))


def indices_to_str(indices: np.ndarray) -> str:
    return "".join(ALPHABET[i] for i in indices)
