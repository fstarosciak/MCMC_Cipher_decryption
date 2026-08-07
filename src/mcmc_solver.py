from __future__ import annotations
import numpy as np


def _precompute_positions(ciphertext: np.ndarray) -> list[np.ndarray]:
    return [np.where(ciphertext == c)[0] for c in range(26)]


def metropolis_hastings(
    ciphertext: np.ndarray,
    log_bigrams: np.ndarray,
    n_iter: int = 10_000,
    initial_key: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    n = len(ciphertext)
    positions = _precompute_positions(ciphertext)

    decrypt_key = (
        initial_key.copy() if initial_key is not None else np.random.permutation(26)
    )
    decrypt_key = decrypt_key.astype(np.int8)
    decoded = decrypt_key[ciphertext]

    current_score = float(log_bigrams[decoded[:-1], decoded[1:]].sum())
    best_key = decrypt_key.copy()
    best_score = current_score
    score_history = [current_score]

    log_rand = np.log(np.random.rand(n_iter))

    for step in range(n_iter):
        i, j = np.random.choice(26, 2, replace=False)

        pos_i = positions[i]
        pos_j = positions[j]

        affected = np.concatenate([pos_i, pos_j])
        if len(affected) == 0:
            continue

        bigram_starts = np.unique(
            np.concatenate([
                affected[affected < n - 1],
                (affected[affected > 0]) - 1,
            ])
        )

        old_contrib = log_bigrams[decoded[bigram_starts], decoded[bigram_starts + 1]].sum()

        old_val_i = decrypt_key[i]
        old_val_j = decrypt_key[j]
        decoded[pos_i] = old_val_j
        decoded[pos_j] = old_val_i

        new_contrib = log_bigrams[decoded[bigram_starts], decoded[bigram_starts + 1]].sum()
        delta = new_contrib - old_contrib

        if log_rand[step] < delta:
            decrypt_key[i] = old_val_j
            decrypt_key[j] = old_val_i
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_key = decrypt_key.copy()
                score_history.append(best_score)
        else:
            decoded[pos_i] = old_val_i
            decoded[pos_j] = old_val_j

    return best_key, best_score, score_history
