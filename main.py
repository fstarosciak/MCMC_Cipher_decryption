import numpy as np
import argparse

from src.corpus import prepare_bigram_matrix, CORPUS_SOURCES
from src.cipher import inverse_key, encrypt, decrypt, key_accuracy, indices_to_str
from src.mcmc_solver import metropolis_hastings
from src import transposition
from src.mcmc_transposition import solve_transposition
from src.experiments import (
    plot_convergence,
    run_monte_carlo_transposition,
    print_results_transposition,
    plot_accuracy_histogram_transposition,
    plot_accuracy_vs_keylength,
    plot_mean_accuracy_vs_keylength,
)

COMMON_N_ITER = 30_000
SEED = 42

TRANS_DEMO_KEY_LEN = 8
TRANS_DEMO_TEXT_LEN = 2000
TRANS_DEMO_RESTARTS = 8

KEY_LENGTHS = [3, 5, 8, 10]
TRANS_TEXT_LEN = 2500
TRANS_N_RUNS = 20
TRANS_N_ITER = 25_000
TRANS_N_RESTARTS = 8

POC_TEXT_LEN = 500


def print_tui_preview(title, original, encrypted, decrypted, sample_len=300):
    width = 100
    print(f"\n{'╔' + '═'*(width-2) + '╗'}")
    print(f"║ {title.center(width-4)} ║")
    print(f"{'╠' + '═'*(width-2) + '╣'}")

    def print_block(label, data):
        text = indices_to_str(data[:sample_len])
        print(f"║ [{label}]".ljust(width-1) + "║")
        for i in range(0, len(text), width - 6):
            line = text[i:i+width-6]
            print(f"║   {line.ljust(width-6)} ║")
        print(f"║{' '.ljust(width-2)}║")

    print_block("TEKST ORYGINALNY", original)
    print_block("SZYFROGRAM (WEJŚCIE)", encrypted)
    print_block("WYNIK DESZYFRACJI (NAJLEPSZY ZNALEZIONY)", decrypted)
    print(f"{'╚' + '═'*(width-2) + '╝'}")


def demo_transposition(test_text, log_bigrams):
    print("\n" + "=" * 60)
    print(f"  DEMONSTRACJA — szyfr kolumnowy"
          f"  (k={TRANS_DEMO_KEY_LEN}, {TRANS_DEMO_TEXT_LEN} liter,"
          f"  {COMMON_N_ITER} iter × {TRANS_DEMO_RESTARTS} restartów)")
    print("=" * 60)

    rng = np.random.default_rng(SEED + 1)
    k = TRANS_DEMO_KEY_LEN
    length = (TRANS_DEMO_TEXT_LEN // k) * k

    start_idx = rng.integers(0, len(test_text) - length)
    plaintext = test_text[start_idx : start_idx + length]

    true_key = rng.permutation(k).astype(np.int8)
    ciphertext = transposition.encrypt(plaintext, true_key)

    print(f"\n[!] SZYFROGRAM (fragment): {indices_to_str(ciphertext[:100])}...")
    print("[*] Rozpoczynam deszyfrowanie MCMC (Parallel Tempering)...")

    found_key, best_score, score_history = solve_transposition(
        ciphertext, log_bigrams,
        key_length=k,
        n_iter=COMMON_N_ITER,
        n_restarts=TRANS_DEMO_RESTARTS,
    )

    recovered = transposition.decrypt(ciphertext, found_key)
    acc_key = transposition.key_accuracy(true_key, found_key)
    acc_text = transposition.text_accuracy(plaintext, recovered)

    print(f"\n  Prawdziwy klucz   : {true_key.tolist()}")
    print(f"  Znaleziony klucz  : {found_key.tolist()}")
    print(f"  Dokładność klucza : {acc_key:.1%}  ({int(acc_key * k)}/{k} pozycji)")
    print(f"  Zgodność liter    : {acc_text:.1%}")
    print(f"  Najlepszy score   : {best_score:.2f}")

    print_tui_preview("WYNIK KOŃCOWY: SZYFR PRZESTAWIENIOWY", plaintext, ciphertext, recovered)

    return score_history


def poc_substitution(test_text, log_bigrams):
    print("\n" + "=" * 60)
    print(f"  PoC — szyfr podstawieniowy  ({POC_TEXT_LEN} liter, {COMMON_N_ITER} iter.)")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    start_idx = rng.integers(0, len(test_text) - POC_TEXT_LEN)
    plaintext = test_text[start_idx : start_idx + POC_TEXT_LEN]

    true_decrypt_key = rng.permutation(26).astype("int8")
    true_encrypt_key = inverse_key(true_decrypt_key)
    ciphertext = encrypt(plaintext, true_encrypt_key)

    print(f"\n[!] SZYFROGRAM (fragment): {indices_to_str(ciphertext[:100])}...")
    print("[*] Rozpoczynam deszyfrowanie MCMC (Metropolis-Hastings)...")

    found_key, best_score, score_history = metropolis_hastings(
        ciphertext, log_bigrams, n_iter=COMMON_N_ITER
    )
    accuracy = key_accuracy(true_decrypt_key, found_key)

    print(f"\n  Dokładność klucza : {accuracy:.1%}  ({int(accuracy * 26)}/26 liter)")
    print(f"  Najlepszy score   : {best_score:.2f}")

    recovered = decrypt(ciphertext, found_key)
    print_tui_preview("WYNIK KOŃCOWY: SZYFR PODSTAWIENIOWY", plaintext, ciphertext, recovered)

    return score_history


def main():
    parser = argparse.ArgumentParser(description="MCMC Cipher Decryption")
    parser.add_argument("--corpus", type=str, default="lalka", choices=list(CORPUS_SOURCES.keys()),
                        help="Wybierz korpus do nauki bigramów")
    args = parser.parse_args()

    np.random.seed(SEED)

    print(f"Przygotowanie macierzy bigramów z korpusu: {args.corpus}...")
    log_bigrams, train_text, test_text = prepare_bigram_matrix(name=args.corpus)

    print("\n" + "=" * 60)
    print(f"  EKSPERYMENTY MONTE CARLO — szyfr przestawieniowy")
    print(f"  N_RUNS={TRANS_N_RUNS}, N_ITER={TRANS_N_ITER}, "
          f"N_RESTARTS={TRANS_N_RESTARTS}, klucze={KEY_LENGTHS}")
    print("=" * 60)

    all_trans_results = []
    for k in KEY_LENGTHS:
        results = run_monte_carlo_transposition(
            test_text, log_bigrams,
            key_length=k,
            text_length=TRANS_TEXT_LEN,
            n_runs=TRANS_N_RUNS,
            n_iter=TRANS_N_ITER,
            n_restarts=TRANS_N_RESTARTS,
        )
        print_results_transposition(results)
        all_trans_results.append(results)

        plot_accuracy_histogram_transposition(results)
        plot_convergence(
            results["score_histories"][:10],
            title=f"trans_{args.corpus}_k{k}",
        )

    plot_accuracy_vs_keylength(all_trans_results)
    plot_mean_accuracy_vs_keylength(all_trans_results)

    print("\n" + "=" * 60)
    print("  DEMONSTRACJA: SZYFR PRZESTAWIENIOWY")
    print("  (Wizualizacja najlepszego wyniku po wszystkich iteracjach)")
    print("=" * 60)

    trans_demo_history = demo_transposition(test_text, log_bigrams)
    plot_convergence(
        [trans_demo_history],
        title=f"trans_demo_{args.corpus}_k{TRANS_DEMO_KEY_LEN}",
    )

    print("\n" + "=" * 60)
    print("  PUNKT ODNIESIENIA: SZYFR PODSTAWIENIOWY")
    print(f"  Używamy tej samej liczby iteracji: {COMMON_N_ITER}")
    print("=" * 60)

    poc_history = poc_substitution(test_text, log_bigrams)
    plot_convergence([poc_history], title=f"poc_substitution_{args.corpus}")

    print("\nGotowe! Wyniki i wykresy zapisano w katalogu results/")


if __name__ == "__main__":
    main()
