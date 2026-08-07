"""Szybki test: czy parametry TRANS_TEXT_LEN=2500, N_ITER=25000, N_RESTARTS=8
dają pełne odszyfrowanie przy k=8 (najtrudniejsza realistyczna wartość w K_LENGTHS)?"""
import numpy as np
import time
from src.corpus import prepare_bigram_matrix
from src.experiments import run_single_experiment_transposition

SEED = 42
K = 10
TEXT_LEN = 2500
N_ITER = 25_000
N_RESTARTS = 8
N_TRIALS = 5

np.random.seed(SEED)
print("Ładowanie korpusu...")
log_bigrams, full_text = prepare_bigram_matrix()

length = (TEXT_LEN // K) * K
plaintext = full_text[:length]

print(f"Test: k={K}, tekst={length}, n_iter={N_ITER}, n_restarts={N_RESTARTS}, trials={N_TRIALS}")
t0 = time.time()
text_accs, key_accs = [], []
for i in range(N_TRIALS):
    ts = time.time()
    r = run_single_experiment_transposition(
        plaintext, log_bigrams,
        key_length=K, n_iter=N_ITER, n_restarts=N_RESTARTS,
    )
    dt = time.time() - ts
    text_accs.append(r["text_acc"])
    key_accs.append(r["key_acc"])
    print(f"  [{i+1}/{N_TRIALS}] text_acc={r['text_acc']:.1%}  key_acc={r['key_acc']:.1%}  "
          f"score={r['best_score']:.1f}  ({dt:.1f}s)")

total = time.time() - t0
print(f"\nŚredni text_acc: {np.mean(text_accs):.1%}")
print(f"Pełne odszyfrowania: {sum(1 for a in text_accs if a==1.0)}/{N_TRIALS}")
print(f"Łączny czas: {total:.1f}s  ({total/N_TRIALS:.1f}s/próba)")
