# Dekrypcja szyfrów klasycznych metodą MCMC

---

## Opis

Projekt implementuje łamanie dwóch klasycznych szyfrów przy użyciu algorytmu
**Metropolis-Hastings** (MCMC):

1. **Szyfr podstawieniowy** (monoalfabetyczny) — przestrzeń kluczy 26! ≈ 4×10²⁶ permutacji.
2. **Szyfr kolumnowy przestawieniowy** — przestrzeń kluczy k! (np. 8! = 40 320 dla k=8).

W obu przypadkach MCMC próbkuje przestrzeń efektywnie, kierując się
**log-prawdopodobieństwem bigramów** obliczonym na korpusie języka polskiego
(„Lalka" Bolesława Prusa, oba tomy, Wolne Lektury — ok. 1.28 mln liter prozy).

Literatura: Chen, R., & Rosenthal, J. S. (2010). *Decrypting Classical Cipher
Text Using Markov Chain Monte Carlo*. Statistics and Computing.

## Wybór korpusu

Użyta jest proza („Lalka"), nie poezja. „Pan Tadeusz" — choć jest kanoniczną
pozycją polskojęzyczną — pisany jest **13-zgłoskowcem**, co zaburza naturalny
rytm bigramów. Proza daje bardziej wiarygodne statystyki językowe.

## Struktura projektu

```
MCMC_Cipher_decryption/
├── main.py                       # punkt wejścia — uruchamia oba demo + eksperymenty
├── requirements.txt
├── src/
│   ├── corpus.py                 # pobieranie korpusu, macierz log-bigramów 26×26
│   ├── cipher.py                 # szyfr podstawieniowy (enc/dec/metryki)
│   ├── mcmc_solver.py            # MH dla podstawieniowego (z optymalizacją delty)
│   ├── transposition.py          # szyfr kolumnowy (enc/dec/metryki)
│   ├── mcmc_transposition.py     # MH dla transpozycji + multi-restart
│   └── experiments.py            # eksperymenty Monte Carlo + wykresy (podstawieniowy)
├── data/                         # tu trafiają pobrane pliki korpusu
└── results/                      # tu trafiają wykresy PNG
```

## Uruchomienie

```bash
pip install -r requirements.txt
python main.py
```

Przy pierwszym uruchomieniu skrypt automatycznie pobierze „Lalkę" (~3 MB).

## Algorytm

### Korpus

Budujemy macierz `log_bigrams[i][j] = log P(j | i)` z ~1.28 mln liter „Lalki".
Polskie diakrytyki są mapowane na ASCII (`ą→a`, `ł→l`, …), wszystkie nie-litery
są ignorowane. Laplace smoothing (+1) unika `log(0)`.

### Szyfr podstawieniowy (MH)

- **Stan:** klucz deszyfrowania = permutacja liter 0–25.
- **Propozycja:** zamień dwie losowo wybrane pozycje w kluczu (symetryczna).
- **Score:** `Σ log_bigrams[decoded[t]][decoded[t+1]]`.
- **Akceptacja:** `min(1, exp(score_new − score_old))`.
- **Optymalizacja:** liczymy tylko **deltę score'u** — bigrams przy zmienionych
  pozycjach (~2/26 tekstu), nie pełne przepisywanie.
- Parametry: 10 000 iteracji, N=100 powtórzeń dla długości [100, 500, 1000, 5000].

### Szyfr kolumnowy — transpozycja (MH)

- **Stan:** klucz = permutacja 0..k-1 (kolejność odczytu kolumn).
- **Propozycja:** swap dwóch pozycji w kluczu.
- **Score:** analogicznie — suma log-bigramów zdekodowanego tekstu.
- **KISS:** pełne przeliczanie score'u na iterację. Swap pozycji w kluczu
  przestawia **całe dwie kolumny** tekstu, więc sprytna delta z szyfru
  podstawieniowego nie daje łatwego zysku. Dla n ~ 1000 i n_iter ~ 10⁴
  pełne przeliczanie zajmuje sekundy.
- **Multi-restart** (`solve_transposition`): delta score'u przy swapie bywa
  rzędu setek, więc zwykłe MH w praktyce **utyka w lokalnym maksimum**
  (efektywnie hill-climbing). Restart z losowego klucza N razy (domyślnie
  N=10) i wybór najlepszego wyniku to prosty i skuteczny sposób.
- Parametry demo: k=8, 1000 liter, 10 000 iter × 10 restartów → 100% klucza.

## Eksperymenty Monte Carlo (szyfr podstawieniowy)

N=100 powtórzeń dla długości tekstu [100, 500, 1000, 5000] liter.
Wyznaczamy: średnią dokładność, odch. std., 95% CI, odsetek pełnych
odszyfrowań.

## Wyniki (wykresy w `results/`)

| Plik | Opis |
|------|------|
| `convergence_*.png` | Krzywa zbieżności najlepszego score'u |
| `accuracy_hist_*.png` | Histogram dokładności klucza (N=100 prób) |
| `accuracy_vs_length_boxplot.png` | Boxplot dokładności vs długość tekstu |
| `mean_accuracy_vs_length.png` | Średnia dokładność z 95% CI vs długość tekstu |
| `convergence_transposition_k8_1000liter.png` | Zbieżność MH dla transpozycji |

## Uwaga o PyMC

PyMC jest zoptymalizowany pod ciągłe modele probabilistyczne (NUTS/HMC).
Przestrzeń permutacji jest dyskretna i kombinatoryczna — algorytm MH
implementujemy ręcznie, zgodnie z literaturą (Chen & Rosenthal, 2010).

## Literatura

- Chen, R., & Rosenthal, J. S. (2010). *Decrypting Classical Cipher Text Using
  Markov Chain Monte Carlo*. Statistics and Computing.
