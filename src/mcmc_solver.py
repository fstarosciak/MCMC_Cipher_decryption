"""
Łamanie szyfru podstawieniowego algorytmem Metropolis-Hastings (MCMC).

Przestrzeń stanów: wszystkie możliwe klucze deszyfrowania — 26! ≈ 4×10^26 permutacji.
Rozkład docelowy: pi(key) ∝ exp( score(key) ), gdzie score = suma log-bigramów
                  (tj. logarytm pseudoprawdopodobieństwa tekstu przy danym kluczu).
Propozycja: zamiana dwóch losowo wybranych pozycji w kluczu deszyfrowania.
            Symetryczna -> współczynnik Hastingsa = 1 -> kryterium czysto Metropolisa.

Uwaga o PyMC: biblioteka PyMC jest zoptymalizowana pod ciągłe modele probabilistyczne
(HMC, NUTS). Przestrzeń permutacji jest dyskretna i kombinatoryczna, dlatego algorytm
MH implementujemy ręcznie — zgodnie z podejściem z literatury (Chen & Rosenthal, 2010).
"""

from __future__ import annotations
import numpy as np


def _precompute_positions(ciphertext: np.ndarray) -> list[np.ndarray]:
    """
    Dla każdej litery c (0-25) zwraca tablicę pozycji w ciphertekście, gdzie ciphertext == c.
    Jednorazowy koszt O(n), potem lookup O(1) per iterację.
    """
    return [np.where(ciphertext == c)[0] for c in range(26)]


def metropolis_hastings(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    n_iter: int = 10_000,
    initial_key: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Algorytm Metropolis-Hastings do łamania szyfru podstawieniowego.

    W każdej iteracji:
      1. Proponuje zamianę dwóch losowych pozycji i, j w kluczu deszyfrowania.
      2. Oblicza zmianę score'u (efektywnie: tylko bigrams przy zmienionych pozycjach).
      3. Akceptuje zmianę z prawdopodobieństwem min(1, exp(delta_score)).
      4. Śledzi najlepszy znaleziony klucz.

    Args:
        ciphertext:   zaszyfrowany tekst jako tablica int, wartości 0-25
        log_bigrams:  macierz 26×26 log-prawdopodobieństw bigramów
        n_iter:       liczba iteracji MH
        initial_key:  startowy klucz deszyfrowania (losowy, jeśli None)

    Returns:
        (best_key, best_score, score_history)
        best_key      — najlepsza znaleziona permutacja deszyfrowania
        best_score    — odpowiadający jej score bigramowy
        score_history — lista najlepszych score'ów w czasie (do wykresów zbieżności)
    """
    n = len(ciphertext)
    positions = _precompute_positions(ciphertext)

    # Inicjalizacja klucza i zdekodowanego tekstu
    decrypt_key = (
        initial_key.copy() if initial_key is not None else np.random.permutation(26)
    )
    decrypt_key = decrypt_key.astype(np.int8)
    decoded = decrypt_key[ciphertext]  # tablica n int8

    current_score = float(log_bigrams[decoded[:-1], decoded[1:]].sum())
    best_key = decrypt_key.copy()
    best_score = current_score
    score_history = [current_score]

    log_rand = np.log(np.random.rand(n_iter))  # prelosuj wszystkie progi akceptacji

    for step in range(n_iter):
        # Propozycja: zamień pozycje i oraz j w kluczu deszyfrowania
        i, j = np.random.choice(26, 2, replace=False)

        pos_i = positions[i]  # indeksy w ciphertekście gdzie stoi litera i
        pos_j = positions[j]  # indeksy w ciphertekście gdzie stoi litera j

        # Znajdź bigrams, na które wpływa zamiana (O(|pos_i| + |pos_j|) ~ O(n/13))
        affected = np.concatenate([pos_i, pos_j])
        if len(affected) == 0:
            continue

        # Pozycje startowe bigramów, które się zmienią
        bigram_starts = np.unique(
            np.concatenate([
                affected[affected < n - 1],
                (affected[affected > 0]) - 1,
            ])
        )

        # Wkład starych bigramów do score'u
        old_contrib = log_bigrams[decoded[bigram_starts], decoded[bigram_starts + 1]].sum()

        # Tymczasowo zaktualizuj decoded (swap wartości liter i <-> j)
        old_val_i = decrypt_key[i]
        old_val_j = decrypt_key[j]
        decoded[pos_i] = old_val_j
        decoded[pos_j] = old_val_i

        # Wkład nowych bigramów
        new_contrib = log_bigrams[decoded[bigram_starts], decoded[bigram_starts + 1]].sum()
        delta = new_contrib - old_contrib

        # Kryterium akceptacji Metropolisa: zaakceptuj jeśli log(U) < delta
        if log_rand[step] < delta:
            # Akceptuj — zaktualizuj klucz i score
            decrypt_key[i] = old_val_j
            decrypt_key[j] = old_val_i
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_key = decrypt_key.copy()
                score_history.append(best_score)
        else:
            # Odrzuć — cofnij zmiany w decoded
            decoded[pos_i] = old_val_i
            decoded[pos_j] = old_val_j

    return best_key, best_score, score_history
