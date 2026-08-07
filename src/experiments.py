from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .cipher import generate_key, inverse_key, encrypt, decrypt, key_accuracy
from .mcmc_solver import metropolis_hastings
from . import transposition as trans
from .mcmc_transposition import solve_transposition

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_HERE, "results")


def run_single_experiment(plaintext, log_bigrams, n_iter=10_000):
    true_decrypt_key = np.random.permutation(26).astype(np.int8)
    true_encrypt_key = inverse_key(true_decrypt_key)
    ciphertext = encrypt(plaintext, true_encrypt_key)
    found_key, best_score, score_history = metropolis_hastings(
        ciphertext, log_bigrams, n_iter=n_iter
    )
    return {
        "accuracy": key_accuracy(true_decrypt_key, found_key),
        "best_score": best_score,
        "score_history": score_history,
    }


def run_monte_carlo(full_text, log_bigrams, text_length=500, n_runs=100, n_iter=10_000):
    if len(full_text) < text_length:
        raise ValueError(f"Tekst ({len(full_text)} liter) krótszy niż text_length={text_length}")
    plaintext = full_text[:text_length]

    plaintext = full_text[:text_length]
    accuracies = []
    score_histories = []

    for _ in tqdm(range(n_runs), desc=f"subst len={text_length:>5}", leave=True):
        result = run_single_experiment(plaintext, log_bigrams, n_iter)
        accuracies.append(result["accuracy"])
        score_histories.append(result["score_history"])

    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std(ddof=1)
    ci_half = 1.96 * std / np.sqrt(n_runs)
    n_perfect = int((accuracies == 1.0).sum())

    return {
        "cipher": "substitution",
        "text_length": text_length,
        "n_runs": n_runs,
        "n_iter": n_iter,
        "accuracies": accuracies,
        "mean_accuracy": mean,
        "std_accuracy": std,
        "ci_95": (max(0.0, mean - ci_half), min(1.0, mean + ci_half)),
        "n_perfect": n_perfect,
        "pct_perfect": n_perfect / n_runs * 100,
        "score_histories": score_histories,
    }


def print_results(r):
    print(f"\n{'─'*55}")
    print(f"  Długość tekstu : {r['text_length']:>5} liter")
    print(f"  Liczba prób    : {r['n_runs']}")
    print(f"  Śr. dok. klucza: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%}")
    print(f"  95% CI         : [{r['ci_95'][0]:.1%}, {r['ci_95'][1]:.1%}]")
    print(f"  Pełne odszyfr. : {r['n_perfect']}/{r['n_runs']} ({r['pct_perfect']:.1f}%)")
    if r.get("cipher") == "transposition":
        print(f"  Śr. dok. liter : {r['mean_text_accuracy']:.1%} ± {r['std_text_accuracy']:.1%}")
        print(f"  95% CI (lit.)  : [{r['ci_95_text'][0]:.1%}, {r['ci_95_text'][1]:.1%}]")
        print(f"  Restartów/próbę: {r['n_restarts']}")


def _savefig(fig, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Zapisano: {path}")


def plot_convergence(score_histories, title="", n_show=10):
    fig, ax = plt.subplots(figsize=(10, 5))
    for hist in score_histories[:n_show]:
        ax.plot(hist, alpha=0.7, linewidth=0.9)
    ax.set_xlabel("Liczba ulepszeń (nowe maximum score)")
    ax.set_ylabel("Najlepszy score (suma log-bigramów)")
    ax.set_title(f"Zbieżność algorytmu MH{(' — ' + title) if title else ''}")
    ax.grid(True, alpha=0.3)
    _savefig(fig, f"convergence_{title.replace(' ', '_')}.png")


def plot_accuracy_histogram(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1 / 52, 1 + 1 / 52, 28)
    ax.hist(r["accuracies"], bins=bins, edgecolor="black", linewidth=0.5,
            color="steelblue" if not is_trans else "teal")
    ax.axvline(r["mean_accuracy"], color="red", linestyle="--", linewidth=1.5,
               label=f"Średnia: {r['mean_accuracy']:.1%}")
    ax.axvspan(r["ci_95"][0], r["ci_95"][1], alpha=0.15, color="red", label="95% CI")

    xlabel = ("Dokładność klucza (odsetek pozycji spośród k)" if is_trans
              else "Dokładność klucza (odsetek trafnych liter spośród 26)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Liczba prób")

    cipher_name = f"transpozycja k={r['key_length']}" if is_trans else "podstawieniowy"
    ax.set_title(
        f"Rozkład dokładności — {cipher_name}, tekst: {r['text_length']} liter (N={r['n_runs']})"
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    suffix = (f"transp_k{r['key_length']}_{r['text_length']}" if is_trans
              else str(r["text_length"]))
    _savefig(fig, f"accuracy_hist_{suffix}.png")


def plot_accuracy_vs_length(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    lengths = [r["text_length"] for r in all_results]
    data = [r["accuracies"] for r in all_results]

    bp = ax.boxplot(data, labels=[str(l) for l in lengths], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    for idx, r in enumerate(all_results, start=1):
        m, lo, hi = r["mean_accuracy"], r["ci_95"][0], r["ci_95"][1]
        ax.plot(idx, m, "ro", markersize=6, zorder=5)
        ax.errorbar(idx, m, yerr=[[m - lo], [hi - m]],
                    fmt="none", color="red", capsize=5, linewidth=1.5)

    ax.set_xlabel("Długość tekstu (liczba liter)")
    ax.set_ylabel("Dokładność klucza")
    ax.set_title(f"Dokładność deszyfrowania vs długość tekstu (N={all_results[0]['n_runs']} prób)")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "accuracy_vs_length_boxplot.png")


def plot_mean_accuracy_vs_length(all_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    lengths = [r["text_length"] for r in all_results]
    means = [r["mean_accuracy"] for r in all_results]
    ci_low = [r["ci_95"][0] for r in all_results]
    ci_high = [r["ci_95"][1] for r in all_results]

    ax.plot(lengths, means, "bo-", linewidth=2, markersize=8, label="Średnia dokładność")
    ax.fill_between(lengths, ci_low, ci_high, alpha=0.2, color="blue", label="95% CI")

    for l, m in zip(lengths, means):
        ax.annotate(f"{m:.1%}", xy=(l, m), xytext=(5, 7), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Długość tekstu (liczba liter)")
    ax.set_ylabel("Średnia dokładność klucza")
    ax.set_title("Wpływ długości tekstu na zbieżność MCMC")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "mean_accuracy_vs_length.png")


def run_single_experiment_transposition(plaintext, log_bigrams, key_length, n_iter=None, n_restarts=None):
    true_key = np.random.permutation(key_length).astype(np.int8)
    ciphertext = trans.encrypt(plaintext, true_key)
    found_key, best_score, score_history = solve_transposition(
        ciphertext, log_bigrams,
        key_length=key_length,
        n_iter=n_iter,
        n_restarts=n_restarts,
    )
    recovered = trans.decrypt(ciphertext, found_key)
    return {
        "key_acc": trans.key_accuracy(true_key, found_key),
        "text_acc": trans.text_accuracy(plaintext, recovered),
        "best_score": best_score,
        "score_history": score_history,
    }


def run_monte_carlo_transposition(full_text, log_bigrams, key_length, text_length=2000,
                                  n_runs=30, n_iter=None, n_restarts=None):
    length = (text_length // key_length) * key_length
    if length == 0:
        raise ValueError(f"text_length={text_length} < key_length={key_length}")
    plaintext = full_text[:length]

    key_accs, text_accs, score_histories = [], [], []

    for _ in tqdm(range(n_runs), desc=f"k={key_length:>2}", leave=True):
        r = run_single_experiment_transposition(plaintext, log_bigrams, key_length, n_iter, n_restarts)
        key_accs.append(r["key_acc"])
        text_accs.append(r["text_acc"])
        score_histories.append(r["score_history"])

    key_accs = np.array(key_accs)
    text_accs = np.array(text_accs)
    mean_t = text_accs.mean()
    std_t = text_accs.std(ddof=1)
    ci_half = 1.96 * std_t / np.sqrt(n_runs)
    n_perfect = int((text_accs == 1.0).sum())

    return {
        "key_length": key_length,
        "text_length": length,
        "n_runs": n_runs,
        "n_iter": n_iter,
        "n_restarts": n_restarts,
        "key_accuracies": key_accs,
        "text_accuracies": text_accs,
        "mean_text_accuracy": mean_t,
        "std_text_accuracy": std_t,
        "ci_95_text": (max(0.0, mean_t - ci_half), min(1.0, mean_t + ci_half)),
        "n_perfect": n_perfect,
        "pct_perfect": n_perfect / n_runs * 100,
        "score_histories": score_histories,
    }


def print_results_transposition(r):
    print(f"\n{'─'*55}")
    print(f"  Długość klucza : {r['key_length']:>3} kolumn  (tekst: {r['text_length']} liter)")
    print(f"  Liczba prób    : {r['n_runs']}")
    print(f"  Śr. zgodność   : {r['mean_text_accuracy']:.1%}  ± {r['std_text_accuracy']:.1%}")
    print(f"  95% CI         : [{r['ci_95_text'][0]:.1%},  {r['ci_95_text'][1]:.1%}]")
    print(f"  Pełne odszyfr. : {r['n_perfect']}/{r['n_runs']}  ({r['pct_perfect']:.1f}%)")


def plot_accuracy_histogram_transposition(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1 / 52, 1 + 1 / 52, 28)
    ax.hist(r["text_accuracies"], bins=bins, edgecolor="black", linewidth=0.5, color="mediumseagreen")
    ax.axvline(r["mean_text_accuracy"], color="red", linestyle="--", linewidth=1.5,
               label=f"Średnia: {r['mean_text_accuracy']:.1%}")
    ax.axvspan(r["ci_95_text"][0], r["ci_95_text"][1], alpha=0.15, color="red", label="95% CI")
    ax.set_xlabel("Zgodność liter (odsetek trafnie odszyfowanych liter)")
    ax.set_ylabel("Liczba prób")
    ax.set_title(f"Rozkład dokładności — szyfr kolumnowy  k={r['key_length']}  (N={r['n_runs']})")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, f"trans_accuracy_hist_k{r['key_length']}.png")


def plot_accuracy_vs_keylength(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    key_lengths = [r["key_length"] for r in all_results]
    data = [r["text_accuracies"] for r in all_results]

    bp = ax.boxplot(data, labels=[str(k) for k in key_lengths], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")
        patch.set_alpha(0.7)

    for idx, r in enumerate(all_results, start=1):
        m = r["mean_text_accuracy"]
        lo, hi = r["ci_95_text"]
        ax.plot(idx, m, "ro", markersize=6, zorder=5)
        ax.errorbar(idx, m, yerr=[[m - lo], [hi - m]],
                    fmt="none", color="red", capsize=5, linewidth=1.5)

    ax.set_xlabel("Długość klucza (liczba kolumn)")
    ax.set_ylabel("Zgodność liter")
    ax.set_title(f"Dokładność deszyfrowania vs długość klucza  (N={all_results[0]['n_runs']} prób)")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "trans_accuracy_vs_keylength_boxplot.png")


def plot_mean_accuracy_vs_keylength(all_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    key_lengths = [r["key_length"] for r in all_results]
    means = [r["mean_text_accuracy"] for r in all_results]
    ci_low = [r["ci_95_text"][0] for r in all_results]
    ci_high = [r["ci_95_text"][1] for r in all_results]

    ax.plot(key_lengths, means, "go-", linewidth=2, markersize=8, label="Średnia zgodność")
    ax.fill_between(key_lengths, ci_low, ci_high, alpha=0.2, color="green", label="95% CI")
    for k, m in zip(key_lengths, means):
        ax.annotate(f"{m:.1%}", xy=(k, m), xytext=(5, 7), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Długość klucza (liczba kolumn)")
    ax.set_ylabel("Średnia zgodność liter")
    ax.set_title("Wpływ długości klucza na zbieżność MCMC  (szyfr kolumnowy)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "trans_mean_accuracy_vs_keylength.png")
