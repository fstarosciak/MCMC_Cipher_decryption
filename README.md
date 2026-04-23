# Dekrypcja szyfrów klasycznych metodą MCMC

---

## Opis

Projekt implementuje łamanie klasycznych szyfrów algorytmem
**Metropolis-Hastings** (MCMC), z naciskiem na **szyfr przestawieniowy
kolumnowy**. Dodatkowo, dla porównania, zaimplementowany jest szyfr
podstawieniowy monoalfabetyczny.

Jako korpus języka polskiego służy **„Lalka" Bolesława Prusa** (oba tomy,
Wolne Lektury — ok. 1.28 mln liter prozy). Z niej budowana jest macierz
log-prawdopodobieństw bigramów, która napędza funkcję score'u w MH.

Literatura: Chen, R., & Rosenthal, J. S. (2010). *Decrypting Classical Cipher
Text Using Markov Chain Monte Carlo*. Statistics and Computing.

### Główny przedmiot badań: szyfr przestawieniowy kolumnowy

- **Przestrzeń kluczy:** `k!` (np. `8! = 40 320` dla k=8, `10! ≈ 3.6·10⁶` dla k=10).
- **Zasada działania:** tekst wpisywany jest wierszami do tabeli o `k` kolumnach,
  a szyfrogram powstaje przez odczyt kolumn w kolejności zadanej kluczem.
- **Dlaczego ten szyfr?** Dyskretna, kombinatoryczna przestrzeń permutacji —
  naturalny test dla MCMC na przestrzeni nieciągłej, gdzie NUTS/HMC się nie nadają.

### Szyfr podstawieniowy (pomocniczo)

- Przestrzeń kluczy 26! ≈ 4·10²⁶ permutacji liter.
- Implementacja głównie jako referencja — pozwala porównać zachowanie MH
  na dwóch różnych przestrzeniach permutacji.

## Wybór korpusu — „Lalka"

Użyta jest proza („Lalka"), nie poezja. „Pan Tadeusz" — choć jest kanoniczną
pozycją polskojęzyczną — pisany jest **13-zgłoskowcem**, co zaburza naturalny
rytm bigramów. Proza daje bardziej wiarygodne statystyki językowe.

Polskie diakrytyki mapowane są na ASCII (`ą→a`, `ł→l`, …), wszystkie
nie-litery są ignorowane. Laplace smoothing (+1) unika `log(0)`.

## Struktura projektu

```
MCMC_Cipher_decryption/
├── main.py                       # punkt wejścia — demo + eksperymenty
├── requirements.txt
├── src/
│   ├── corpus.py                 # pobieranie „Lalki", macierz log-bigramów 26×26
│   ├── transposition.py          # szyfr kolumnowy przestawieniowy (enc/dec/metryki)
│   ├── mcmc_transposition.py     # MH dla transpozycji + multi-restart
│   ├── cipher.py                 # szyfr podstawieniowy (enc/dec/metryki)
│   ├── mcmc_solver.py            # MH dla podstawieniowego (z optymalizacją delty)
│   └── experiments.py            # eksperymenty Monte Carlo + wykresy
├── data/                         # pobrane pliki korpusu
└── results/                      # wykresy PNG
```

## Uruchomienie

```bash
pip install -r requirements.txt
python main.py
```

Przy pierwszym uruchomieniu skrypt automatycznie pobierze „Lalkę" (~3 MB).

## Algorytm

### Szyfr przestawieniowy kolumnowy (MH) — główny

- **Stan:** klucz = permutacja `0..k-1` (kolejność odczytu kolumn).
- **Propozycja:** swap dwóch pozycji w kluczu (symetryczna).
- **Score:** `Σ log_bigrams[decoded[t]][decoded[t+1]]` — log-wiarygodność
  zdekodowanego tekstu względem bigramów „Lalki".
- **Akceptacja:** `min(1, exp(score_new − score_old))`.
- **KISS — pełne przeliczanie score'u:** swap pozycji w kluczu przestawia
  **całe dwie kolumny** tekstu, więc „sprytna delta" znana z szyfru
  podstawieniowego nie daje łatwego zysku. Dla `n ~ 1000` i `n_iter ~ 10⁴`
  pełne przeliczanie zajmuje sekundy.
- **Multi-restart** (`solve_transposition`): delta score'u przy swapie
  bywa rzędu setek, więc zwykłe MH w praktyce **utyka w lokalnym maksimum**
  (efektywnie hill-climbing). Restart z losowego klucza `N` razy
  (domyślnie `N=10`) i wybór najlepszego wyniku to prosty i skuteczny
  sposób obejścia.
- Parametry demo: `k=8`, 1000 liter, 10 000 iter × 10 restartów → 100% klucza.

### Szyfr podstawieniowy (MH) — referencyjnie

- **Stan:** klucz deszyfrowania = permutacja liter 0–25.
- **Propozycja:** zamień dwie losowo wybrane pozycje w kluczu.
- **Score:** jak wyżej — suma log-bigramów zdekodowanego tekstu.
- **Optymalizacja delty:** liczymy tylko zmianę score'u przy zmienionych
  pozycjach (~2/26 tekstu), nie pełne przepisywanie.
- Parametry: 10 000 iteracji, N=100 powtórzeń dla długości [100, 500, 1000, 5000].

## Eksperymenty Monte Carlo

Dla obu szyfrów: N powtórzeń, wyznaczamy średnią dokładność, odch. std.,
95% CI, odsetek pełnych odszyfrowań.

- **Transpozycja:** zmienna długość klucza `k ∈ {3, 5, 8, 10}`.
- **Podstawieniowy:** zmienna długość tekstu `[100, 500, 1000, 5000]` liter.

## Wyniki (wykresy w `results/`)

### Transpozycja — główne

| Plik | Opis |
|------|------|
| `convergence_trans_k{3,5,8,10}.png` | Zbieżność MH dla różnych długości klucza |
| `trans_accuracy_hist_k{3,5,8,10}.png` | Histogram dokładności klucza |
| `trans_accuracy_vs_keylength_boxplot.png` | Boxplot dokładności vs `k` |
| `trans_mean_accuracy_vs_keylength.png` | Średnia dokładność z 95% CI vs `k` |
| `convergence_trans_demo_k8_2000liter.png` | Demo k=8, 2000 liter |

### Podstawieniowy — referencyjnie

| Plik | Opis |
|------|------|
| `convergence_*liter.png` | Krzywa zbieżności najlepszego score'u |
| `accuracy_hist_*.png` | Histogram dokładności klucza (N=100 prób) |
| `accuracy_vs_length_boxplot.png` | Boxplot dokładności vs długość tekstu |
| `mean_accuracy_vs_length.png` | Średnia dokładność z 95% CI vs długość tekstu |

## Uwaga o PyMC

PyMC jest zoptymalizowany pod ciągłe modele probabilistyczne (NUTS/HMC).
Przestrzeń permutacji jest dyskretna i kombinatoryczna — algorytm MH
implementujemy ręcznie, zgodnie z literaturą (Chen & Rosenthal, 2010).

## Literatura

- Chen, R., & Rosenthal, J. S. (2010). *Decrypting Classical Cipher Text Using
  Markov Chain Monte Carlo*. Statistics and Computing.
