"""
Łamanie szyfru kolumnowego metodą Metropolis-Hastings (MCMC).

Przestrzeń stanów: permutacje długości k (k! możliwości — przy k=10 to 3.6·10^6,
więc MH daje realny zysk względem brute force już od k ≈ 8).
Rozkład docelowy: pi(key) ∝ exp(score(key)), gdzie score = suma log-bigramów
                  na zdekodowanym tekście.
Propozycja: zamiana dwóch losowych pozycji w kluczu (symetryczna -> kryterium Metropolisa).

KISS: na każdej iteracji przeliczamy pełny score zdekodowanego tekstu.
W odróżnieniu od szyfru podstawieniowego, swap pozycji w kluczu przestawia
CAŁE dwie kolumny, więc „sprytna delta" z mcmc_solver.py nie daje łatwego zysku.
Dla n ~ 1000 i n_iter ~ 10^4 pełne przeliczanie zajmuje sekundy.
"""

from __future__ import annotations
import numpy as np

from .transposition import decrypt


def _score(decoded: np.ndarray, log_bigrams: np.ndarray) -> float:
    """Suma log-prawdopodobieństw bigramów w zdekodowanym tekście."""
    return float(log_bigrams[decoded[:-1], decoded[1:]].sum())


def metropolis_hastings_transposition(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int,
    n_iter: int = 10_000,
    initial_key: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Metropolis-Hastings dla szyfru kolumnowego przy znanej długości klucza.

    W każdej iteracji:
      1. Proponuje swap dwóch losowych pozycji w kluczu.
      2. Deszyfruje ciphertext bieżącym kluczem i liczy score bigramowy.
      3. Akceptuje swap z prawdopodobieństwem min(1, exp(delta)).
      4. Zapamiętuje najlepszy dotąd klucz.

    Args:
        ciphertext:  tablica indeksów 0-25; długość musi dzielić się przez key_length
        log_bigrams: macierz 26×26 log-prawdopodobieństw bigramów
        key_length:  długość klucza (liczba kolumn)
        n_iter:      liczba iteracji MH
        initial_key: startowy klucz (losowy, jeśli None)

    Returns:
        (best_key, best_score, score_history) — analogicznie do mcmc_solver.py
    """
    if len(ciphertext) % key_length != 0:
        raise ValueError(
            f"Długość ciphertextu ({len(ciphertext)}) nie dzieli się przez "
            f"key_length ({key_length})."
        )

    key = (
        initial_key.copy().astype(np.int8)
        if initial_key is not None
        else np.random.permutation(key_length).astype(np.int8)
    )
    current_score = _score(decrypt(ciphertext, key), log_bigrams)
    best_key = key.copy()
    best_score = current_score
    score_history = [current_score]

    log_rand = np.log(np.random.rand(n_iter))

    for step in range(n_iter):
        i, j = np.random.choice(key_length, 2, replace=False)
        key[i], key[j] = key[j], key[i]

        new_score = _score(decrypt(ciphertext, key), log_bigrams)
        delta = new_score - current_score

        if log_rand[step] < delta:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_key = key.copy()
                score_history.append(best_score)
        else:
            # odrzuć — wróć do poprzedniego klucza
            key[i], key[j] = key[j], key[i]

    return best_key, best_score, score_history


def solve_transposition(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int,
    n_iter: int = 10_000,
    n_restarts: int = 10,
) -> tuple[np.ndarray, float, list[float]]:
    """
    MH z wieloma restartami (multi-start). Przy transpozycji delta score'u
    przy swapie kolumn bywa rzędu setek, więc zwykłe MH szybko utyka w lokalnym
    maksimum (efektywnie hill-climbing). Restart z losowego punktu to prosty
    i skuteczny sposób na pokonanie tego problemu.

    Zwraca wynik restartu o najwyższym score'ze oraz jego score_history
    (wygodne do wykresu zbieżności „najlepszego przebiegu").
    """
    best_key = None
    best_score = -np.inf
    best_history: list[float] = []
    for _ in range(n_restarts):
        key, score, history = metropolis_hastings_transposition(
            ciphertext, log_bigrams, key_length, n_iter=n_iter
        )
        if score > best_score:
            best_key, best_score, best_history = key, score, history
    return best_key, best_score, best_history
