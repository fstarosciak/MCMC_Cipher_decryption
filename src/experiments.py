"""
Eksperymenty Monte Carlo: wielokrotne uruchamianie MH z losowymi kluczami startowymi.
Analiza statystyczna i wizualizacje wyników.
Obsługuje dwa szyfry:
 - podstawieniowy (monoalfabetyczny) — run_monte_carlo()
 - kolumnowy (transpozycyjny)        — run_monte_carlo_transposition()
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .cipher import generate_key, inverse_key, encrypt, decrypt, key_accuracy
from .mcmc_solver import metropolis_hastings
from . import transposition
from .mcmc_transposition import solve_transposition

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_HERE, "results")


# ─── Pojedynczy eksperyment: podstawieniowy ───────────────────────────────────

def run_single_experiment(
    plaintext: np.ndarray,
    log_bigrams: np.ndarray,
    n_iter: int = 10_000,
) -> dict:
    """
    Losuje klucz szyfrowania, szyfruje tekst, uruchamia MH i mierzy dokładność.
    Returns:
        dict: accuracy (float), best_score (float), score_history (list)
    """
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


# ─── Pojedynczy eksperyment: transpozycja ─────────────────────────────────────

def run_single_experiment_transposition(
    plaintext: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int,
    n_iter: int = 10_000,
    n_restarts: int = 10,
) -> dict:
    """
    Logika analogiczna do demo_transposition():
    1. Losuje klucz kolumnowy (permutację 0..key_length-1).
    2. Szyfruje plaintext szyfrem kolumnowym.
    3. Uruchamia solve_transposition() (MH z multi-restart).
    4. Mierzy dokładność klucza i zgodność liter.
    Tekst jest przycinany do wielokrotności key_length — tak samo jak w demo.
    Returns:
        dict: key_accuracy, text_accuracy, best_score, score_history
    """
    # Przytnij do wielokrotności key_length (identycznie jak w demo_transposition)
    length = (len(plaintext) // key_length) * key_length
    plaintext = plaintext[:length]

    true_key = np.random.permutation(key_length).astype(np.int8)
    ciphertext = transposition.encrypt(plaintext, true_key)

    found_key, best_score, score_history = solve_transposition(
        ciphertext, log_bigrams,
        key_length=key_length,
        n_iter=n_iter,
        n_restarts=n_restarts,
    )
    recovered = transposition.decrypt(ciphertext, found_key)

    return {
        "key_accuracy": transposition.key_accuracy(true_key, found_key),
        "text_accuracy": transposition.text_accuracy(plaintext, recovered),
        "best_score": best_score,
        "score_history": score_history,
    }


# ─── Eksperyment Monte Carlo: podstawieniowy ─────────────────────────────────

def run_monte_carlo(
    full_text: np.ndarray,
    log_bigrams: np.ndarray,
    text_length: int = 500,
    n_runs: int = 100,
    n_iter: int = 10_000,
) -> dict:
    """
    Powtarza łamanie szyfru podstawieniowego n_runs razy dla danej długości tekstu.
    Tekst jest zawsze taki sam (pierwsze text_length liter korpusu),
    zmieniają się tylko losowe klucze szyfrowania i punkty startowe MH.
    Returns:
        dict ze statystykami i surową listą dokładności (do wykresów)
    """
    if len(full_text) < text_length:
        raise ValueError(
            f"Tekst ({len(full_text)} liter) krótszy niż text_length={text_length}"
        )

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


# ─── Eksperyment Monte Carlo: transpozycja ───────────────────────────────────

def run_monte_carlo_transposition(
    full_text: np.ndarray,
    log_bigrams: np.ndarray,
    key_length: int = 8,
    text_length: int = 1000,
    n_runs: int = 100,
    n_iter: int = 10_000,
    n_restarts: int = 10,
) -> dict:
    """
    Powtarza łamanie szyfru kolumnowego n_runs razy.
    Każda próba:
      - losuje nowy klucz kolumnowy,
      - szyfruje ten sam fragment tekstu (pierwsze text_length liter, przycięte do k),
      - uruchamia solve_transposition() z multi-restart (identycznie jak demo_transposition),
      - mierzy key_accuracy i text_accuracy.
    Args:
        full_text:   pełny korpus jako tablica int8
        log_bigrams: macierz log-bigramów 26×26
        key_length:  liczba kolumn (k) — przestrzeń kluczy to k!
        text_length: żądana długość tekstu (przycięta do wielokrotności k)
        n_runs:      liczba powtórzeń Monte Carlo
        n_iter:      liczba iteracji MH na jeden restart
        n_restarts:  liczba restartów na jeden przebieg (multi-restart)
    Returns:
        dict ze statystykami — analogicznie do run_monte_carlo(),
        plus pola specyficzne dla transpozycji: key_length, n_restarts,
        actual_text_length (po przycięciu), text_accuracies.
    """
    # Rzeczywista długość po przycięciu do wielokrotności k
    actual_length = (text_length // key_length) * key_length
    if actual_length == 0:
        raise ValueError(
            f"text_length={text_length} zbyt krótkie dla key_length={key_length}"
        )
    if len(full_text) < actual_length:
        raise ValueError(
            f"Korpus ({len(full_text)} liter) krótszy niż actual_length={actual_length}"
        )

    plaintext = full_text[:actual_length]
    key_accuracies = []
    text_accuracies = []
    score_histories = []

    desc = f"transp k={key_length} len={actual_length:>5}"
    for _ in tqdm(range(n_runs), desc=desc, leave=True):
        result = run_single_experiment_transposition(
            plaintext, log_bigrams,
            key_length=key_length,
            n_iter=n_iter,
            n_restarts=n_restarts,
        )
        key_accuracies.append(result["key_accuracy"])
        text_accuracies.append(result["text_accuracy"])
        score_histories.append(result["score_history"])

    key_accuracies = np.array(key_accuracies)
    text_accuracies = np.array(text_accuracies)

    mean_k = key_accuracies.mean()
    std_k = key_accuracies.std(ddof=1)
    ci_half_k = 1.96 * std_k / np.sqrt(n_runs)
    n_perfect_k = int((key_accuracies == 1.0).sum())

    mean_t = text_accuracies.mean()
    std_t = text_accuracies.std(ddof=1)
    ci_half_t = 1.96 * std_t / np.sqrt(n_runs)

    return {
        "cipher": "transposition",
        "key_length": key_length,
        "text_length": text_length,
        "actual_text_length": actual_length,
        "n_runs": n_runs,
        "n_iter": n_iter,
        "n_restarts": n_restarts,
        # dokładność klucza
        "accuracies": key_accuracies,  # alias dla kompatybilności z plot_*
        "mean_accuracy": mean_k,
        "std_accuracy": std_k,
        "ci_95": (max(0.0, mean_k - ci_half_k), min(1.0, mean_k + ci_half_k)),
        "n_perfect": n_perfect_k,
        "pct_perfect": n_perfect_k / n_runs * 100,
        # dokładność liter (dodatkowa metryka)
        "text_accuracies": text_accuracies,
        "mean_text_accuracy": mean_t,
        "std_text_accuracy": std_t,
        "ci_95_text": (max(0.0, mean_t - ci_half_t), min(1.0, mean_t + ci_half_t)),
        "score_histories": score_histories,
    }


# ─── Drukowanie wyników ───────────────────────────────────────────────────────

def print_results(r: dict) -> None:
    """Wypisuje podsumowanie jednego eksperymentu (oba szyfry)."""
    cipher_label = {
        "substitution": "Szyfr podstawieniowy",
        "transposition": "Szyfr transpozycyjny",
    }.get(r.get("cipher", ""), "Szyfr")

    print(f"\n{'─' * 55}")
    print(f"  {cipher_label}")
    print(f"  Długość tekstu : {r['text_length']:>5} liter", end="")
    if r.get("cipher") == "transposition":
        print(f" (k={r['key_length']}, faktyczna: {r['actual_text_length']})", end="")
    print()
    print(f"  Liczba prób    : {r['n_runs']}")
    print(f"  Śr. dok. klucza: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%}")
    print(f"  95% CI         : [{r['ci_95'][0]:.1%}, {r['ci_95'][1]:.1%}]")
    print(f"  Pełne odszyfr. : {r['n_perfect']}/{r['n_runs']} ({r['pct_perfect']:.1f}%)")
    if r.get("cipher") == "transposition":
        print(f"  Śr. dok. liter : {r['mean_text_accuracy']:.1%} ± {r['std_text_accuracy']:.1%}")
        print(f"  95% CI (lit.)  : [{r['ci_95_text'][0]:.1%}, {r['ci_95_text'][1]:.1%}]")
        print(f"  Restartów/próbę: {r['n_restarts']}")


# ─── Wykresy: wspólne ────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, filename: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Zapisano: {path}")


def plot_convergence(score_histories: list[list], title: str = "", n_show: int = 10) -> None:
    """
    Krzywa zbieżności: najlepszy score w czasie dla kilku przebiegów MH.
    Oś X = numer znalezionego ulepszenia (nie numer iteracji).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for hist in score_histories[:n_show]:
        ax.plot(hist, alpha=0.7, linewidth=0.9)
    ax.set_xlabel("Liczba ulepszeń (nowe maximum score)")
    ax.set_ylabel("Najlepszy score (suma log-bigramów)")
    ax.set_title(f"Zbieżność algorytmu MH{(' — ' + title) if title else ''}")
    ax.grid(True, alpha=0.3)
    _savefig(fig, f"convergence_{title.replace(' ', '_')}.png")


def plot_accuracy_histogram(r: dict) -> None:
    """Histogram dokładności klucza dla N prób (działa dla obu szyfrów)."""
    cipher = r.get("cipher", "substitution")
    is_trans = cipher == "transposition"

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


def plot_accuracy_vs_length(all_results: list[dict]) -> None:
    """Boxplot dokładności dla różnych długości tekstu + punkty średnich z CI."""
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


def plot_mean_accuracy_vs_length(all_results: list[dict]) -> None:
    """Wykres liniowy: średnia dokładność z 95% CI w funkcji długości tekstu."""
    fig, ax = plt.subplots(figsize=(8, 5))
    lengths = [r["text_length"] for r in all_results]
    means = [r["mean_accuracy"] for r in all_results]
    ci_low = [r["ci_95"][0] for r in all_results]
    ci_high = [r["ci_95"][1] for r in all_results]

    ax.plot(lengths, means, "bo-", linewidth=2, markersize=8, label="Średnia dokładność")
    ax.fill_between(lengths, ci_low, ci_high, alpha=0.2, color="blue", label="95% CI")

    for l, m in zip(lengths, means):
        ax.annotate(f"{m:.1%}", xy=(l, m), xytext=(5, 7),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel("Długość tekstu (liczba liter)")
    ax.set_ylabel("Średnia dokładność klucza")
    ax.set_title("Wpływ długości tekstu na zbieżność MCMC")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "mean_accuracy_vs_length.png")


# ─── Wykresy porównawcze (podstawieniowy vs transpozycja) ────────────────────

def plot_comparison_accuracy_vs_length(
    subst_results: list[dict],
    trans_results: list[dict],
) -> None:
    """
    Porównanie średniej dokładności klucza obu szyfrów w funkcji długości tekstu.
    Oba zestawy muszą mieć te same text_length (lub porównywalne).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    s_lengths = [r["text_length"] for r in subst_results]
    s_means = [r["mean_accuracy"] for r in subst_results]
    s_ci_low = [r["ci_95"][0] for r in subst_results]
    s_ci_high = [r["ci_95"][1] for r in subst_results]

    t_lengths = [r["actual_text_length"] for r in trans_results]
    t_means = [r["mean_accuracy"] for r in trans_results]
    t_ci_low = [r["ci_95"][0] for r in trans_results]
    t_ci_high = [r["ci_95"][1] for r in trans_results]

    k = trans_results[0]["key_length"] if trans_results else "?"

    ax.plot(s_lengths, s_means, "bo-", linewidth=2, markersize=8,
            label="Podstawieniowy (26!)")
    ax.fill_between(s_lengths, s_ci_low, s_ci_high, alpha=0.15, color="blue")

    ax.plot(t_lengths, t_means, "gs-", linewidth=2, markersize=8,
            label=f"Transpozycja k={k} ({k}!)")
    ax.fill_between(t_lengths, t_ci_low, t_ci_high, alpha=0.15, color="green")

    for l, m in zip(s_lengths, s_means):
        ax.annotate(f"{m:.0%}", xy=(l, m), xytext=(-18, 6),
                    textcoords="offset points", fontsize=8, color="blue")
    for l, m in zip(t_lengths, t_means):
        ax.annotate(f"{m:.0%}", xy=(l, m), xytext=(5, -14),
                    textcoords="offset points", fontsize=8, color="green")

    ax.set_xlabel("Długość tekstu (liczba liter)")
    ax.set_ylabel("Średnia dokładność klucza")
    ax.set_title(
        f"Porównanie szyfrów — dokładność vs długość tekstu (N={subst_results[0]['n_runs']})"
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "comparison_accuracy_vs_length.png")


def plot_comparison_boxplot(
    subst_results: list[dict],
    trans_results: list[dict],
) -> None:
    """
    Boxplot obok siebie: dokładność klucza dla obu szyfrów, każda długość tekstu.
    Grupy na osi X = długości tekstu; w każdej grupie dwa boxy (subst / transp).
    """
    lengths = sorted(set(
        [r["text_length"] for r in subst_results] +
        [r["text_length"] for r in trans_results]
    ))
    subst_map = {r["text_length"]: r for r in subst_results}
    trans_map = {r["text_length"]: r for r in trans_results}

    n_groups = len(lengths)
    fig, ax = plt.subplots(figsize=(max(10, 3 * n_groups), 6))

    positions_s, positions_t = [], []
    data_s, data_t = [], []
    x_ticks, x_labels = [], []

    gap = 0.9   # odległość między grupami
    w = 0.35    # szerokość jednego boksa

    for i, length in enumerate(lengths):
        center = i * (2 * w + gap + 0.2)
        ps = center - w / 2
        pt = center + w / 2
        if length in subst_map:
            positions_s.append(ps)
            data_s.append(subst_map[length]["accuracies"])
        if length in trans_map:
            positions_t.append(pt)
            data_t.append(trans_map[length]["accuracies"])
        x_ticks.append(center)
        x_labels.append(str(length))

    def _bp(data, positions, color, label):
        bp = ax.boxplot(
            data, positions=positions, widths=w,
            patch_artist=True, manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for element in ["whiskers", "caps", "medians", "fliers"]:
            for line in bp[element]:
                line.set_color(color)
        # Phantom handle dla legendy
        ax.plot([], [], color=color, linewidth=6, alpha=0.65, label=label)

    k = trans_results[0]["key_length"] if trans_results else "?"
    _bp(data_s, positions_s, "steelblue", "Podstawieniowy (26!)")
    _bp(data_t, positions_t, "seagreen", f"Transpozycja k={k} ({k}!)")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Długość tekstu (liczba liter)")
    ax.set_ylabel("Dokładność klucza")
    ax.set_title(
        f"Porównanie szyfrów — boxplot dokładności (N={subst_results[0]['n_runs']} prób)"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "comparison_boxplot.png")
