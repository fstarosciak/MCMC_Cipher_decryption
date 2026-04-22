"""
Łamanie szyfru kolumnowego przez MCMC (Parallel Tempering).
"""

from __future__ import annotations
import numpy as np

from .transposition import decrypt


def _score(decoded: np.ndarray, log_bigrams: np.ndarray) -> float:
    # Sumujemy logarytmy prawdopodobieństw występowania par liter obok siebie
    return float(log_bigrams[decoded[:-1], decoded[1:]].sum())


def _propose(key: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Losowa modyfikacja klucza: swap, odwrócenie fragmentu albo przesunięcie elementu
    k = len(key)
    new_key = key.copy()
    u = rng.random()

    if u < 0.7 or k < 4:
        # Zamiana dwóch kolumn
        i, j = rng.choice(k, 2, replace=False)
        new_key[i], new_key[j] = new_key[j], new_key[i]
    elif u < 0.9:
        # Odwrócenie kolejności w losowym bloku
        i, j = sorted(rng.choice(k, 2, replace=False))
        new_key[i:j + 1] = new_key[i:j + 1][::-1]
    else:
        # Wycięcie jednej kolumny i wsadzenie jej w inne miejsce
        i, j = rng.choice(k, 2, replace=False)
        val = new_key[i]
        new_key = np.delete(new_key, i)
        new_key = np.insert(new_key, j if j < i else j - 1, val)

    return new_key.astype(np.int8)


def parallel_tempering(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int,
    n_iter: int = 20_000,
    n_chains: int = 6,
    t_min: float = 0.4,
    t_max: float = 6.0,
    swap_every: int = 40,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    # Kilka łańcuchów w różnych temperaturach. Zimne szukają lokalnie, gorące skaczą po całym rozwiązaniu.
    if rng is None:
        rng = np.random.default_rng()

    # Rozkład temperatur (skala geometryczna)
    temperatures = t_min * (t_max / t_min) ** (
        np.arange(n_chains) / max(n_chains - 1, 1)
    )

    keys = [rng.permutation(key_length).astype(np.int8) for _ in range(n_chains)]
    scores = [_score(decrypt(ciphertext, k), log_bigrams) for k in keys]

    best_key = keys[0].copy()
    best_score = scores[0]
    best_history = [best_score]

    for step in range(n_iter):
        for c in range(n_chains):
            # Standardowy krok Metropolis-Hastings dla każdego łańcucha
            proposed = _propose(keys[c], rng)
            new_score = _score(decrypt(ciphertext, proposed), log_bigrams)
            delta = (new_score - scores[c]) / temperatures[c]
            
            if np.log(rng.random()) < delta:
                keys[c] = proposed
                scores[c] = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_key = proposed.copy()
                    best_history.append(best_score)

        # Co jakiś czas próbujemy zamienić stany między sąsiednimi łańcuchami
        if step % swap_every == 0 and step > 0:
            for c in range(n_chains - 1):
                log_alpha = (scores[c] - scores[c + 1]) * (
                    1.0 / temperatures[c + 1] - 1.0 / temperatures[c]
                )
                if np.log(rng.random()) < log_alpha:
                    keys[c], keys[c + 1] = keys[c + 1], keys[c]
                    scores[c], scores[c + 1] = scores[c + 1], scores[c]

    return best_key, best_score, best_history


def metropolis_hastings_transposition(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int,
    n_iter: int = 20_000,
    initial_key: np.ndarray | None = None,
    t_start: float = 3.0,
    t_end: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    # Prostsza wersja algorytmu z jednym łańcuchem i stopniowym chłodzeniem
    if len(ciphertext) % key_length != 0:
        raise ValueError("Długość tekstu nie pasuje do długości klucza.")
    
    if rng is None:
        rng = np.random.default_rng()

    key = (
        initial_key.copy().astype(np.int8)
        if initial_key is not None
        else rng.permutation(key_length).astype(np.int8)
    )
    current_score = _score(decrypt(ciphertext, key), log_bigrams)
    best_key = key.copy()
    best_score = current_score
    score_history = [current_score]

    cooling = (t_end / t_start) ** (1.0 / max(n_iter - 1, 1))
    temperature = t_start
    log_rand = np.log(rng.random(n_iter))

    for step in range(n_iter):
        proposed = _propose(key, rng)
        new_score = _score(decrypt(ciphertext, proposed), log_bigrams)
        delta = (new_score - current_score) / temperature

        if log_rand[step] < delta:
            key = proposed
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_key = key.copy()
                score_history.append(best_score)

        temperature *= cooling

    return best_key, best_score, score_history


def solve_transposition(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int,
    n_iter: int | None = None,
    n_restarts: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    # Główny punkt wejścia - odpala PT kilka razy i wybiera najlepszy wynik ze wszystkich prób
    if n_iter is None:
        n_iter = max(6_000, 1_500 * key_length)
    if n_restarts is None:
        n_restarts = max(3, key_length // 2)

    rng = np.random.default_rng(seed)
    best_key: np.ndarray | None = None
    best_score = -np.inf
    best_history: list[float] = []

    for _ in range(n_restarts):
        key, score, history = parallel_tempering(
            ciphertext, log_bigrams, key_length,
            n_iter=n_iter, rng=rng,
        )
        if score > best_score:
            best_key, best_score, best_history = key, score, history

    assert best_key is not None
    return best_key, best_score, best_history
