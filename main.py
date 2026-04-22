"""
Główny skrypt projektu: Łamanie szyfru podstawieniowego metodą MCMC.

Projekt: Modelowanie Monte Carlo — Temat nr 13
Autorzy: Michał Czajkowski, Filip Starościak

Schemat działania:
  1. Pobierz korpus „Lalka" (Project Gutenberg) i zbuduj macierz bigramów.
  2. Demonstracja — jeden pełny cykl: szyfrowanie → MH → weryfikacja.
  3. Eksperymenty Monte Carlo — N=100 prób dla długości tekstu [100, 500, 1000, 5000].
  4. Wykresy: zbieżność MH, histogram dokładności, boxplot vs długość.
"""

import numpy as np
import string

from src.corpus import prepare_bigram_matrix
from src.cipher import inverse_key, encrypt, decrypt, key_accuracy, indices_to_str
from src.mcmc_solver import metropolis_hastings
from src import transposition
from src.mcmc_transposition import solve_transposition
from src.experiments import (
    run_monte_carlo,
    print_results,
    plot_convergence,
    plot_accuracy_histogram,
    plot_accuracy_vs_length,
    plot_mean_accuracy_vs_length,
)

# ─── Parametry szyfru przestawieniowego ──────────────────────────────────────
TRANSPOSITION_KEY_LEN = 8      # długość klucza (liczba kolumn)
TRANSPOSITION_TEXT_LEN = 1000  # długość tekstu w demo
TRANSPOSITION_N_ITER = 10_000  # iteracji MH na jeden restart
TRANSPOSITION_RESTARTS = 10    # restartów (MH szybko utyka w lokalnym maksimum)

# ─── Parametry ────────────────────────────────────────────────────────────────

TEXT_LENGTHS = [100, 500, 1000, 5000]   # długości tekstu do eksperymentów [litery]
N_RUNS      = 100                        # liczba powtórzeń Monte Carlo
N_ITER      = 10_000                     # iteracji MH na jeden przebieg
SEED        = 42

# ──────────────────────────────────────────────────────────────────────────────


def demo(full_text: np.ndarray, log_bigrams: np.ndarray) -> list[float]:
    """
    Demonstracja jednego pełnego cyklu szyfrowania i deszyfrowania (500 liter).
    Wypisuje próbkę oryginału i odszyfowanego tekstu.
    """
    print("\n" + "=" * 60)
    print("  DEMONSTRACJA — jeden przebieg MH  (500 liter, 10 000 iter.)")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    plaintext = full_text[:500]

    true_decrypt_key = rng.permutation(26).astype("int8")
    true_encrypt_key = inverse_key(true_decrypt_key)
    ciphertext = encrypt(plaintext, true_encrypt_key)

    found_key, best_score, score_history = metropolis_hastings(
        ciphertext, log_bigrams, n_iter=N_ITER
    )
    accuracy = key_accuracy(true_decrypt_key, found_key)

    print(f"  Dokładność klucza : {accuracy:.1%}  ({int(accuracy * 26)}/26 liter)")
    print(f"  Najlepszy score   : {best_score:.2f}")

    recovered = decrypt(ciphertext, found_key)
    sample = 100
    print(f"\n  Oryginał   (pierwsze {sample} liter): {indices_to_str(plaintext[:sample])}")
    print(f"  Odszyfr.   (pierwsze {sample} liter): {indices_to_str(recovered[:sample])}")

    n_match = np.sum(recovered[:sample] == plaintext[:sample])
    print(f"  Trafne litery w próbce: {n_match}/{sample}")

    return score_history


def demo_transposition(full_text: np.ndarray, log_bigrams: np.ndarray) -> list[float]:
    """
    Demonstracja łamania szyfru kolumnowego (transpozycyjnego) metodą MH.
    Szyfrujemy fragment korpusu znanym kluczem, potem MH próbuje go odtworzyć.
    """
    print("\n" + "=" * 60)
    print(f"  DEMONSTRACJA — szyfr kolumnowy  "
          f"(k={TRANSPOSITION_KEY_LEN}, {TRANSPOSITION_TEXT_LEN} liter, "
          f"{TRANSPOSITION_N_ITER} iter × {TRANSPOSITION_RESTARTS} restartów)")
    print("=" * 60)

    rng = np.random.default_rng(SEED + 1)
    k = TRANSPOSITION_KEY_LEN
    # Tekst przycinamy do wielokrotności k, żeby demo było deterministyczne
    length = (TRANSPOSITION_TEXT_LEN // k) * k
    plaintext = full_text[:length]

    true_key = rng.permutation(k).astype(np.int8)
    ciphertext = transposition.encrypt(plaintext, true_key)

    found_key, best_score, score_history = solve_transposition(
        ciphertext, log_bigrams,
        key_length=k,
        n_iter=TRANSPOSITION_N_ITER,
        n_restarts=TRANSPOSITION_RESTARTS,
    )
    recovered = transposition.decrypt(ciphertext, found_key)

    acc_key = transposition.key_accuracy(true_key, found_key)
    acc_text = transposition.text_accuracy(plaintext, recovered)

    print(f"  Prawdziwy klucz    : {true_key.tolist()}")
    print(f"  Znaleziony klucz   : {found_key.tolist()}")
    print(f"  Dokładność klucza  : {acc_key:.1%}  ({int(acc_key * k)}/{k} pozycji)")
    print(f"  Zgodność liter     : {acc_text:.1%}")
    print(f"  Najlepszy score    : {best_score:.2f}")

    sample = 100
    print(f"\n  Oryginał   (pierwsze {sample}): {indices_to_str(plaintext[:sample])}")
    print(f"  Odszyfr.   (pierwsze {sample}): {indices_to_str(recovered[:sample])}")

    n_match = int(np.sum(recovered[:sample] == plaintext[:sample]))
    print(f"  Trafne litery w próbce: {n_match}/{sample}")

    return score_history


def main() -> None:
    np.random.seed(SEED)

    # 1. Korpus i macierz bigramów
    print("Przygotowanie macierzy bigramów z korpusu...")
    log_bigrams, full_text = prepare_bigram_matrix()
    print(f"Macierz bigramów: {log_bigrams.shape}, min={log_bigrams.min():.2f}, "
          f"max={log_bigrams.max():.2f}")

    # 2. Demonstracja — szyfr podstawieniowy
    demo_history = demo(full_text, log_bigrams)
    plot_convergence([demo_history], title="demo_500liter")

    # 2b. Demonstracja — szyfr kolumnowy (transpozycyjny)
    trans_history = demo_transposition(full_text, log_bigrams)
    plot_convergence(
        [trans_history],
        title=f"transposition_k{TRANSPOSITION_KEY_LEN}_{TRANSPOSITION_TEXT_LEN}liter",
    )

    # 3. Eksperymenty Monte Carlo
    print("\n" + "=" * 60)
    print(f"  EKSPERYMENTY MONTE CARLO")
    print(f"  N_RUNS={N_RUNS}, N_ITER={N_ITER}, długości={TEXT_LENGTHS}")
    print("=" * 60)

    all_results = []
    for length in TEXT_LENGTHS:
        results = run_monte_carlo(
            full_text, log_bigrams,
            text_length=length,
            n_runs=N_RUNS,
            n_iter=N_ITER,
        )
        print_results(results)
        all_results.append(results)

        plot_accuracy_histogram(results)
        plot_convergence(
            results["score_histories"][:10],
            title=f"{length}liter",
        )

    # 4. Wykresy zbiorcze
    plot_accuracy_vs_length(all_results)
    plot_mean_accuracy_vs_length(all_results)

    print("\nGotowe! Wyniki i wykresy zapisano w katalogu  results/")


if __name__ == "__main__":
    main()
