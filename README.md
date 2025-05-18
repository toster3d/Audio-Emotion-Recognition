# Rozpoznawanie Emocji w Nagraniach Audio

## Wstęp

Celem projektu jest stworzenie zaawansowanego modelu rozpoznającego emocje w nagraniach audio. Automatyczne rozpoznawanie emocji w mowie (**Speech Emotion Recognition - SER**) znajduje szerokie zastosowanie w interakcji człowiek-komputer, analizie sentymentu w rozmowach (np. w call center), diagnostyce medycznej czy personalizacji usług cyfrowych.

Projekt koncentruje się na badaniu wpływu różnorodnych reprezentacji akustycznych oraz technik uczenia zespołowego na skuteczność rozpoznawania emocji przy użyciu konwolucyjnych sieci neuronowych.

## Zbiór Danych: nEMO

Projekt wykorzystuje polskojęzyczny zbiór danych **nEMO** (National Emotional Speech Corpus).

Charakterystyka zbioru nEMO:

*   **Źródło:** Został stworzony przez Center for Artificial Intelligence na Uniwersytecie im. Adama Mickiewicza w Poznaniu.
*   **Zawartość:** Nagrania mowy w języku polskim.
*   **Emocje:** Nagrania są etykietowane jedną z sześciu podstawowych emocji: **szczęście (happiness), złość (anger), strach (fear), smutek (sadness), zaskoczenie (surprised)** oraz **neutralny (neutral)**.
*   **Struktura:** Zbiór zawiera m.in. ID pliku, ścieżkę do pliku audio, etykietę emocji, surowy i znormalizowany tekst wypowiedzi, ID mówcy, płeć i wiek mówcy.
*   **Rozmiar:** Zbiór treningowy zawiera 4480 nagrań (dane z przeglądarki zbioru). Warto zaznaczyć, że zbiór charakteryzuje się zróżnicowaną długością nagrań (od 0.87 do 5.66 sekundy).

Szczegółowe informacje o zbiorze dostępne są na platformie Hugging Face: [https://huggingface.co/datasets/amu-cai/nEMO](https://huggingface.co/datasets/amu-cai/nEMO)

## Przetwarzanie Danych Audio i Augmentacja

Proces przygotowania danych audio obejmuje ekstrakcję cech oraz zaawansowane techniki augmentacji:

*   **Ekstrakcja Cech:** Z surowych nagrań audio wyodrębnianych jest **jedenaście różnych reprezentacji akustycznych (cech)**, które mają uchwycić odmienne aspekty sygnału mowy. Należą do nich:
    *   **Spektrogramy Mel'a:** Popularne wizualne reprezentacje energii sygnału, z osią częstotliwości przeskalowaną do skali Mel.
    *   **Spektrogramy Liniowe:** Standardowe spektrogramy przedstawiające rozkład energii w dziedzinie częstotliwości i czasu.
    *   **MFCC (Mel-frequency cepstral coefficients):** Współczynniki często wykorzystywane w przetwarzaniu mowy, opisujące obwiednię widma.
    *   **Chromagramy:** Reprezentują rozkład energii chromatycznej w czasie, użyteczne do analizy harmonii.
    *   **Spectral Contrast:** Mierzy różnicę między wartościami szczytowymi i dolinami w widmie, co może pomóc w rozróżnianiu harmonicznych i szumu.
    *   **Zero Crossing Rate (ZCR):** Prosta cecha opisująca częstotliwość zmiany znaku sygnału, związana z rytmem i perkusyjnością.
    *   **Root Mean Square Energy (RMS):** Mierzy energię sygnału w czasie, użyteczna do segmentacji i analizy głośności.
    *   **Tempogram:** Reprezentacja tempa muzycznego sygnału audio w czasie, adaptowalna do analizy rytmu mowy.
    *   **Tonnetz:** Reprezentacja chromatyczności w 6-wymiarowej przestrzeni, związana z harmonią toniczną.
    *   **Delta MFCC i Delta Tempogram:** Dynamiczne cechy, będące pierwszymi pochodnymi MFCC i Tempogramu w czasie, uchwytujące zmiany cech bazowych.

*   **Buforowanie Cech (Caching):** Aby przyspieszyć iteracyjne eksperymenty i uniknąć wielokrotnego przeliczania cech, przetworzone reprezentacje audio są **buforowane na dysku**. Umożliwia to szybkie wczytywanie danych w kolejnych przebiegach eksperymentów.

*   **Zaawansowana Augmentacja Danych:** Projekt implementuje rozbudowany, modularny system augmentacji danych audio, wykorzystujący **wzorzec fabryki (factory pattern)**. Dzięki temu, dla każdej z jedenastu reprezentacji cech audio, stosowane są dedykowane strategie transformacji. Obejmują one techniki takie jak:
    *   Dodawanie szumu (dostosowane poziomy dla różnych cech).
    *   Maskowanie w dziedzinie czasu i częstotliwości (specyficzne dla reprezentacji 2D, np. spektrogramów).
    *   Przesunięcia w czasie.
    *   Maskowanie harmoniczne (dla cech harmonicznych jak chroma, tonnetz).
    *   Augmentacje specyficzne dla cech jednowymiarowych (np. dodawanie szumu do ZCR, RMS).
    Celem augmentacji jest zwiększenie różnorodności zbioru treningowego, redukcja ryzyka przeuczenia i poprawa generalizacji modelu na nowe, nieznane dane.

*   **Podział Danych:** Dane są standardowo dzielone na zbiory treningowe, walidacyjne i testowe w celu monitorowania procesu uczenia i obiektywnej oceny końcowej wydajności modelu.

## Architektura Modeli

Projekt eksploruje dwie główne strategie modelowania:

*   **Indywidualne Modele AudioResNet:** Dla każdej z jedenastu wyekstrahowanych reprezentacji audio trenowany jest osobny model konwolucyjnej sieci neuronowej. Główną architekturą jest **AudioResNet**, bazujący na klasycznym **ResNet18** z biblioteki PyTorch (`torchvision.models.resnet18`), zaadaptowany do przetwarzania danych audio (np. przez zmianę pierwszych warstw konwolucyjnych, aby akceptowały jednokanałowe dane wejściowe reprezentujące cechy audio).

*   **Model Ensemble (Ensemble Model):** Dodatkowo rozwijany jest model ensemble, który łączy predykcje z wybranych indywidualnych modeli AudioResNet. Obecna implementacja modelu ensemble (**`WeightedEnsembleModel`**) opiera się na łączeniu predykcji (prawdopodobieństw klas) z trzech modeli trenowanych na reprezentacjach **melspektrogram, MFCC i chroma** przy użyciu **ważonej sumy**. Wagi poszczególnych modeli składowych w ensemble mogą być optymalizowane, co pozwala na dynamiczne dostosowanie wpływu każdej reprezentacji na ostateczną predykcję. Model ensemble może również zawierać parametr *temperatury* do kalibracji rozkładu prawdopodobieństw oraz regularyzację L1 na wagach w celu promowania rzadkości.

## Trening i Optymalizacja

Proces treningu modeli indywidualnych i ensemble jest zoptymalizowany i monitorowany:

*   **Trening Indywidualnych Modeli:** Modele AudioResNet są trenowane niezależnie dla każdej reprezentacji cech w celu oceny ich indywidualnej skuteczności. Proces ten jest zaimplementowany m.in. w notatniku `src/ResNet_porownanie.ipynb`.

*   **Trening i Optymalizacja Modelu Ensemble:** Model ensemble jest trenowany w celu optymalizacji wag, z jakimi łączone są predykcje modeli składowych. Wykorzystywana jest w tym celu biblioteka **Optuna**, która automatyzuje proces wyszukiwania najlepszej kombinacji wag, minimalizując straty walidacyjne modelu ensemble. Ten proces jest realizowany m.in. w notatniku `src/ResNet_ensemble.ipynb` oraz skryptach pomocniczych (np. `src/helpers/ensemble_trainer.py`).

*   **Śledzenie Eksperymentów:** Przebieg wszystkich eksperymentów, użyte hiperparametry, metryki wydajności (np. straty, dokładność, F1-score) oraz wygenerowane artefakty (np. wytrenowane modele, wykresy) są śledzone i zarządzane za pomocą narzędzia **MLflow**. MLflow zapewnia centralne repozytorium dla wyników eksperymentów, ułatwiając ich porównywanie, analizę i replikację.

*   **Wczesne Zatrzymywanie (Early Stopping):** Zaimplementowano mechanizm wczesnego zatrzymywania, który monitoruje straty na zbiorze walidacyjnym i automatycznie przerywa trening, jeśli straty przestają maleć przez określoną liczbę epok (patience). Pomaga to zapobiegać przeuczeniu.

*   **Regularyzacja:** Stosowane są techniki regularyzacji, takie jak Dropout w modelach AudioResNet oraz opcjonalna regularyzacja L1 na wagach w modelu ensemble, w celu poprawy generalizacji i zapobiegania nadmiernemu dopasowaniu do danych treningowych.

## Ewaluacja Modelu

Skuteczność modeli jest oceniana przy użyciu standardowych metryk klasyfikacji, wyliczanych na zbiorze testowym. Kluczowe metryki obejmują:

*   **Accuracy (Dokładność):** Ogólny odsetek poprawnie sklasyfikowanych próbek.
*   **F1-score:** Średnia harmoniczna precyzji i kompletności (recall), często wyliczana dla poszczególnych klas emocji oraz uśredniona (macro/weighted average) w celu uwzględnienia potencjalnej nierównowagi klas.
*   **Macierz Pomyłek (Confusion Matrix):** Wizualna reprezentacja wyników klasyfikacji, pokazująca liczbę poprawnych i błędnych klasyfikacji dla każdej pary klas emocji. Pomaga zidentyfikować, które emocje są mylone przez model.

Analiza wyników dla poszczególnych reprezentacji i porównanie ich z modelem ensemble są kluczowe dla oceny, które cechy są najbardziej informatywne i czy uczenie zespołowe przynosi poprawę.

## Struktura Projektu i Workflow Eksperymentalny

Projekt został zaprojektowany w sposób modularny, zgodny z dobrymi praktykami inżynierii uczenia maszynowego, co ułatwia rozbudowę i utrzymanie. Poszczególne etapy pipeline'u ML są wydzielone do dedykowanych modułów i skryptów. Główny workflow eksperymentalny, w szczególności dla porównania indywidualnych modeli na różnych cechach, jest zaimplementowany w notatniku `src/ResNet_porownanie.ipynb`. Obejmuje on:

1.  **Ładowanie Danych:** Wczytywanie surowego zbioru danych nEMO.
2.  **Ekstrakcja i Buforowanie Cech:** Wyodrębnianie jedenastu reprezentacji audio i opcjonalne zapisywanie ich na dysku (caching).
3.  **Definicja i Trening Modeli:** Iteracyjne tworzenie i trening modelu AudioResNet dla każdej z wyekstrahowanych reprezentacji.
4.  **Ewaluacja i Zapis Wyników:** Ocena trenowanego modelu na zbiorze testowym i zapis metryk (accuracy, F1-score, macierz pomyłek) do plików.
5.  **Analiza i Wizualizacja:** Odczytanie i agregacja wyników ze wszystkich eksperymentów (np. za pomocą funkcji z `src/helpers/data_proccesing.py`), generowanie wykresów porównawczych i macierzy pomyłek w celu analizy skuteczności różnych cech.

Rozwój i optymalizacja modelu ensemble odbywa się w notatniku `src/ResNet_ensemble.ipynb` oraz powiązanych skryptach, skupiając się na ładowaniu indywidualnych modeli, optymalizacji wag ensemble za pomocą Optuna i ewaluacji końcowej.

## Eksport Modelu do ONNX

Model ensemble, po wytrenowaniu i zoptymalizowaniu, może zostać wyeksportowany do formatu **ONNX (Open Neural Network Exchange)**. ONNX to otwarty format wymiany modeli uczenia maszynowego, który umożliwia wdrożenie wytrenowanych modeli w szerokiej gamie środowisk i platform, niezależnie od frameworku, w którym zostały oryginalnie stworzone (np. PyTorch, TensorFlow). Eksport do ONNX jest kluczowy dla integracji modelu z systemami produkcyjnymi, aplikacjami mobilnymi czy rozwiązaniami typu Edge AI, często dzięki zoptymalizowanym środowiskom uruchomieniowym, takim jak **ONNX Runtime**.

Proces eksportu zaimplementowany w projekcie (m.in. w skryptach w `src/export_scripts/`) obejmuje:

1.  **Konwersję:** Przekształcenie modelu PyTorch (`WeightedEnsembleModel`) do formatu ONNX.
2.  **Optymalizację:** Zastosowanie narzędzi optymalizacyjnych (np. `onnxoptimizer`, `onnxsim`) w celu redukcji rozmiaru modelu i zwiększenia wydajności inferencji.
3.  **Kwantyzację (Opcjonalnie):** Dodatkowa optymalizacja polegająca na zmniejszeniu precyzji wag i aktywacji (np. z 32-bitowych liczb zmiennoprzecinkowych do 8-bitowych liczb całkowitych), co znacząco redukuje rozmiar modelu i przyspiesza inferencję, kosztem potencjalnie niewielkiej utraty dokładności. Jest to szczególnie przydatne w środowiskach z ograniczonymi zasobami obliczeniowymi.
4.  **Weryfikację:** Przeprowadzenie testów zgodności, porównujących wyjścia modelu ONNX z wyjściami oryginalnego modelu PyTorch dla tych samych danych wejściowych, aby upewnić się, że proces eksportu przebiegł poprawnie i model ONNX działa zgodnie z oczekiwaniami. Skrypty testowe (np. `src/export_scripts/test_onnx_model.py`, `src/export_scripts/test_quantization.py`) umożliwiają automatyczne porównania i weryfikację.

Dzięki temu pipeline'owi, wytrenowany model ensemble jest gotowy do łatwego wdrożenia.

## Technologie

Projekt wykorzystuje szereg narzędzi i bibliotek, obejmujących przetwarzanie audio, uczenie maszynowe, zarządzanie eksperymentami i wdrożenie modeli:

*   **Język programowania:** Python
*   **Biblioteki do przetwarzania i analizy audio:** `librosa`, `soundfile`, `soxr` (do resamplingu), `pyAudioAnalysis` (potencjalnie do ekstrakcji cech, choć większość cech jest prawdopodobnie liczona przez `librosa`/implementacje własne)
*   **Ładowanie danych:** `datasets` (prawdopodobnie do obsługi zbioru nEMO z Hugging Face), `tensorflow-io` (do wydajnego ładowania danych audio)
*   **Biblioteka uczenia maszynowego:** `PyTorch` (do budowy i treningu modeli CNN)
*   **Format wymiany modeli ML:** `ONNX`
*   **Środowisko uruchomieniowe ONNX:** `ONNX Runtime`
*   **Narzędzia do optymalizacji modeli ONNX:** `onnxsim`, `onnxoptimizer`
*   **Narzędzia do wizualizacji:** `matplotlib`, `seaborn`, `librosa.display`, `plotly`
*   **Narzędzia do zarządzania eksperymentami:** `MLflow`, `TensorBoard`
*   **Narzędzia do optymalizacji hiperparametrów:** `Optuna`
*   **Narzędzia do analizy modeli (opcjonalnie):** `shap`
*   **Środowisko pracy:** Jupyter Notebook, Google Colab (sugerowane przez użycie GPU)
*   **Narzędzia do wersjonowania kodu:** Git, GitHub
*   **Zarządzanie zależnościami:** `poetry` (sugerowane przez plik `pyproject.toml`)
*   **Linting/Formatowanie kodu:** `ruff`

## Przykładowe Komendy / Użycie

Poniżej przedstawiono przykładowe komendy demonstrujące użycie wybranych skryptów w projekcie. Przed uruchomieniem upewnij się, że wszystkie zależności są zainstalowane (np. za pomocą `poetry install`) i że posiadasz odpowiednie pliki modeli.

*   **Eksport modelu ensemble do ONNX:**
    ```bash
    python src/export_scripts/export_to_onnx.py --model_path <ścieżka_do_modelu_pytorch> --output_dir <ścieżka_wyjściowa_dla_onnx>
    ```
    (Model ensemble PyTorch musi być wcześniej wytrenowany i zapisany)

*   **Testowanie modelu ONNX na pliku audio:**
    ```bash
    python src/export_scripts/test_onnx_model.py --model_path <ścieżka_do_pliku_onnx> --audio_path <ścieżka_do_pliku_audio.wav>
    ```
    (Wymaga podania ścieżki do wyeksportowanego modelu ONNX i pliku audio do testu)

*   **Kwantyzacja modelu ONNX:**
    ```bash
    python src/export_scripts/test_quantization.py --model_path <ścieżka_do_pliku_onnx> --output_dir <ścieżka_wyjściowa_dla_skwantyzowanego_onnx>
    ```
    (Tworzy skwantyzowaną wersję modelu ONNX)

*   **Porównanie wyników modelu ONNX (oryginalnego lub skwantyzowanego) z modelem PyTorch:**
    ```bash
    python src/export_scripts/test_quantization.py --compare --model_path <ścieżka_do_pliku_onnx> --pytorch_model_path <ścieżka_do_modelu_pytorch>
    ```
    (Pomaga zweryfikować, czy eksport lub kwantyzacja nie wpłynęły znacząco na precyzję predykcji)

## Możliwości Rozbudowy i Przyszłe Kierunki

Architektura projektu została zaprojektowana z myślą o łatwej rozbudowie i eksperymentowaniu. Możliwe kierunki dalszego rozwoju obejmują:

*   **Eksploracja Nowych Architektur Modeli:** Testowanie innych wariantów ResNet (np. ResNet34, ResNet50) lub zupełnie innych architektur CNN (np. z warstwami 1D dla sekwencji cech, modele oparte o transformery, np. Conformer) czy modeli specyficznych dla audio.
*   **Badanie Dodatkowych Reprezentacji Audio:** Implementacja i ocena nowych typów cech akustycznych lub zaawansowanych technik ekstrakcji cech.
*   **Testowanie Alternatywnych Metod Augmentacji:** Wprowadzenie i ewaluacja nowych strategii augmentacji danych, wykraczających poza obecny zestaw, w celu dalszego zwiększenia odporności modelu na szum i wariancje w danych.
*   **Eksploracja Nowych Sposobów Łączenia Cech/Modeli:** Badanie alternatywnych technik uczenia zespołowego (np. stacking, boosting) lub innych metod łączenia różnych reprezentacji audio, na przykład poprzez podawanie ich jako wielokanałowego wejścia do pojedynczej sieci konwolucyjnej na wczesnym etapie (Early Fusion).
*   **Dalsza Optymalizacja:** Strojenie hiperparametrów zarówno modeli indywidualnych, jak i modelu ensemble (np. przy użyciu zaawansowanych strategii Optuna).
*   **Ulepszenie Pipeline'u MLOps:** Rozwój automatyzacji całego pipeline'u ML (trening, ewaluacja, wersjonowanie danych i modeli, wdrożenie) przy użyciu narzędzi takich jak DVC, Jenkins/GitHub Actions, Kubernetes.
*   **Wdrożenie w Czasie Rzeczywistym:** Rozwój rozwiązań umożliwiających inferencję modelu w czasie rzeczywistym (np. z wykorzystaniem mikroserwisów, serverless, WebAssembly dla przeglądarek).

Projekt stanowi solidną podstawę do kontynuowania badań nad rozpoznawaniem emocji w audio i może być rozwijany zarówno w celach akademickich, jak i komercyjnych.
