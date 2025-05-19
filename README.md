# Rozpoznawanie Emocji w Nagraniach Audio

## Wstęp

Celem niniejszego projektu jest opracowanie i ewaluacja zaawansowanego systemu do automatycznego rozpoznawania emocji w mowie (Speech Emotion Recognition - SER) w języku polskim. Systemy SER znajdują szerokie zastosowanie, między innymi w interakcjach człowiek-komputer, analizie sentymentu w centrach obsługi klienta, wspomaganiu diagnostyki medycznej oraz personalizacji usług cyfrowych.

Projekt koncentruje się na zbadaniu wpływu różnorodnych reprezentacji akustycznych sygnału mowy oraz zastosowaniu technik uczenia zespołowego (ensemble learning) na skuteczność rozpoznawania emocji przy użyciu głębokich konwolucyjnych sieci neuronowych (CNN).

## Zbiór Danych: nEMO

Do realizacji projektu wykorzystano polskojęzyczny zbiór danych **nEMO** (National Emotional Speech Corpus), udostępniony przez Centrum Sztucznej Inteligencji Uniwersytetu im. Adama Mickiewicza w Poznaniu.

Charakterystyka zbioru nEMO:
*   **Źródło:** Center for Artificial Intelligence, Uniwersytet im. Adama Mickiewicza w Poznaniu.
*   **Zawartość:** Nagrania mowy w języku polskim.
*   **Emocje:** Nagrania są etykietowane jedną z sześciu podstawowych emocji: **złość (anger), strach (fear), szczęście (happiness), smutek (sadness), zaskoczenie (surprised)** oraz stan **neutralny (neutral)**.
*   **Struktura:** Zbiór zawiera metadane takie jak ID pliku, ścieżkę do pliku audio, etykietę emocji, transkrypcję (surową i znormalizowaną), ID mówcy, płeć oraz wiek mówcy.
*   **Rozmiar:** Zbiór treningowy, wykorzystywany w projekcie, zawiera 4480 nagrań. Długość nagrań jest zróżnicowana, waha się od około 0.87 do 5.66 sekundy.

Szczegółowe informacje o zbiorze nEMO dostępne są na platformie Hugging Face: [https://huggingface.co/datasets/amu-cai/nEMO](https://huggingface.co/datasets/amu-cai/nEMO)

## Przetwarzanie Danych Audio i Augmentacja

Proces przygotowania danych audio obejmuje ekstrakcję cech, ich buforowanie oraz zaawansowane techniki augmentacji danych.

### Ekstrakcja Cech Akustycznych

Z surowych nagrań audio wyodrębnianych jest **jedenaście różnych reprezentacji akustycznych**, mających na celu uchwycenie odmiennych aspektów sygnału mowy, istotnych dla rozpoznawania emocji. Wszystkie nagrania są przycinane lub dopełniane zerami do stałej długości (domyślnie 3 sekundy, konfigurowalne w `src/config.py` jako `MAX_LENGTH`) przed ekstrakcją cech. Implementacja ekstrakcji znajduje się głównie w notatniku `src/ResNet_porownanie.ipynb` (funkcja `extract_features`).

Wykorzystywane cechy to:
1.  **Spektrogramy Mel'a (Melspectrogram):** Wizualne reprezentacje energii sygnału w dziedzinie czasu i częstotliwości, z osią częstotliwości przeskalowaną logarytmicznie do skali Mel, która lepiej oddaje ludzką percepcję wysokości dźwięku.
2.  **Spektrogramy Liniowe (Spectrogram):** Standardowe spektrogramy (STFT) przedstawiające rozkład energii sygnału.
3.  **MFCC (Mel-frequency cepstral coefficients):** Współczynniki cepstralne w skali Mel, szeroko stosowane w przetwarzaniu mowy, opisujące kształt obwiedni widma.
4.  **Chromagramy (Chroma):** Reprezentują rozkład energii w dwunastu klasach chromatycznych (półtonach) w czasie, użyteczne do analizy zawartości tonalnej i harmonicznej.
5.  **Kontrast Spektralny (Spectral Contrast):** Mierzy różnicę energii między szczytami a dolinami w poszczególnych pasmach widma, co może pomagać w rozróżnianiu dźwięków harmonicznych od szumowych.
6.  **Zero Crossing Rate (ZCR):** Częstotliwość, z jaką sygnał przechodzi przez zero. Cecha ta jest związana z tonem i "perkusyjnością" dźwięku. W projekcie, cecha 1D ZCR jest rozszerzana do formatu 2D (kompatybilnego z wejściem CNN) poprzez specyficzne skalowanie i powielenie w funkcji `extract_features`.
7.  **Root Mean Square Energy (RMS):** Mierzy energię (głośność) sygnału w krótkich oknach czasowych. Podobnie jak ZCR, cecha 1D RMS jest przekształcana do formatu 2D w funkcji `extract_features`.
8.  **Tempogram:** Reprezentacja tempa i rytmu sygnału audio w czasie.
9.  **Tonnetz (Tonal Centroids):** Sześciowymiarowa reprezentacja centrów tonalnych, opisująca relacje harmoniczne w muzyce, adaptowalna do analizy intonacji mowy.
10. **Delta MFCC (ΔMFCC):** Pierwsze pochodne współczynników MFCC w czasie, opisujące dynamikę zmian w widmie.
11. **Delta Tempogram (ΔTempogram):** Pierwsze pochodne tempogramu, uchwytujące zmiany w rytmie.

### Buforowanie Przetworzonych Cech (Caching)

Aby przyspieszyć iteracyjne eksperymenty i uniknąć wielokrotnego, czasochłonnego przeliczania cech, przetworzone reprezentacje audio dla każdej z jedenastu cech są **buforowane na dysku** (serializowane za pomocą `pickle`). Pliki cache zapisywane są w katalogu `src/processed_features/` (konfigurowalne jako `PROCESSED_FEATURES_DIR` w `src/config.py`) z unikalnym identyfikatorem opartym na parametrach ekstrakcji. Funkcjonalność ta jest zaimplementowana w funkcji `process_dataset` w notatniku `src/ResNet_porownanie.ipynb`.

### Zaawansowana Augmentacja Danych

Projekt implementuje rozbudowany, modularny system augmentacji danych audio, zdefiniowany w `src/helpers/augment_for_all_types.py`. Wykorzystuje on **wzorzec projektowy fabryki (factory pattern)** oraz strategii, aby dla każdej z jedenastu reprezentacji cech audio stosować dedykowane techniki transformacji. Augmentacja jest stosowana wyłącznie do zbioru treningowego (w ramach klasy `AugmentedAudioDataset` używanej w `src/ResNet_porownanie.ipynb`) w celu zwiększenia jego różnorodności, redukcji ryzyka przeuczenia i poprawy zdolności generalizacji modelu na nowe, nieznane dane. Przykładowe techniki augmentacji (zależne od typu cechy) obejmują:
*   Dodawanie szumu (z różnymi poziomami dla różnych cech).
*   Maskowanie w dziedzinie czasu i częstotliwości (dla reprezentacji 2D, np. spektrogramów).
*   Przesunięcia w czasie.
*   Maskowanie harmoniczne (dla cech takich jak chroma, tonnetz).
*   Augmentacje specyficzne dla cech jednowymiarowych (np. dodawanie szumu do ZCR, RMS przed ich rozszerzeniem).

### Przygotowanie Danych do Modelu

Przed podaniem do modelu, dane są odpowiednio przygotowywane:
*   **Podział Danych:** W głównym workflow porównywania cech (notatnik `src/ResNet_porownanie.ipynb`), dane dla każdej cechy są dzielone na zbiory: treningowy, walidacyjny i testowy (domyślnie w proporcjach 64%-16%-20% całości danych, konfigurowalne przez `TEST_SPLIT` w `src/config.py` dla podziału testowego, a następnie kolejny podział zbioru treningowego na treningowy i walidacyjny) przy użyciu stratyfikacji względem etykiet emocji. Funkcja `process_dataset` w tym notatniku zawiera również logikę przygotowania podziałów dla walidacji krzyżowej (domyślnie 5-fold, `CV_FOLDS` w `config.py`), która jest wykorzystywana np. podczas optymalizacji wag modelu ensemble.
*   **Normalizacja Danych:** Cechy mogą być normalizowane per próbka podczas ekstrakcji oraz/lub cały zbiór danych może być normalizowany globalnie (np. przez odjęcie średniej i podzielenie przez odchylenie standardowe), co jest kontrolowane flagami w funkcji `process_dataset`.
*   **Obsługa Batchy Danych:** Do ładowania danych w batchach używane są klasy `torch.utils.data.DataLoader`. Dla niektórych cech (szczególnie ZCR, RMS po ich rozszerzeniu) oraz dla modelu ensemble, stosowane są niestandardowe funkcje `collate_fn` (np. `audio_collate_fn` w `src/ResNet_porownanie.ipynb` oraz `ensemble_collate_fn` w `src/helpers/ensemble_dataset.py`), które obsługują dopełnianie tensorów do wspólnego rozmiaru w ramach batcha, co jest niezbędne dla architektur CNN.

## Architektura Modeli

Projekt eksploruje dwie główne strategie modelowania: trening indywidualnych modeli dla każdej reprezentacji audio oraz budowę modelu ensemble łączącego predykcje z wybranych modeli indywidualnych.

### Indywidualne Modele AudioResNet

Dla każdej z jedenastu wyekstrahowanych reprezentacji akustycznych trenowany jest osobny model konwolucyjnej sieci neuronowej. Główną architekturą bazową jest **AudioResNet**, zdefiniowany w `src/helpers/resnet_model_definition.py`. Model ten opiera się na popularnej architekturze **ResNet18** z biblioteki `torchvision.models`, która została zaadaptowana do przetwarzania danych audio:
*   Pierwsza warstwa konwolucyjna (`conv1`) została zmodyfikowana, aby akceptować jednokanałowe dane wejściowe (reprezentujące cechy audio w skali szarości) zamiast trójkanałowych obrazów RGB.
*   Oryginalna warstwa w pełni połączona (fully connected layer - `fc`) na końcu ResNet18 została zastąpiona warstwą `nn.Identity()`, a następnie dodano nową warstwę `nn.Dropout` (z siłą określoną przez `DROPOUT_RATE` w `src/config.py`) oraz nową warstwę w pełni połączoną, mapującą cechy wyekstrahowane przez ResNet na liczbę klas emocji (sześć).
Wagi modelu są inicjalizowane przy użyciu standardowych technik (Kaiming dla warstw konwolucyjnych, Xavier dla warstw liniowych), co jest zaimplementowane w konstruktorze klasy `AudioResNet`.

### Model Ensemble (WeightedEnsembleModel)

Dodatkowo rozwijany jest model ensemble, który łączy predykcje z wybranych indywidualnych modeli AudioResNet. W projekcie zaimplementowano **`WeightedEnsembleModel`** (zdefiniowany w `src/helpers/ensemble_model.py`), który agreguje predykcje (prawdopodobieństwa klas) z trzech modeli AudioResNet trenowanych na reprezentacjach: **melspektrogram, MFCC oraz chroma**. Wybór tych cech wynikał z ich dobrych wyników indywidualnych oraz komplementarności informacji, jakie niosą. Modele bazowe są ładowane automatycznie na podstawie ich najlepszych zapisanych wersji z eksperymentów porównawczych (funkcja `auto_configure_ensemble` w `src/ResNet_ensemble.ipynb`).

Kluczowe cechy `WeightedEnsembleModel`:
*   **Ważona Suma Predykcji:** Predykcje z modeli składowych są łączone za pomocą ważonej sumy. Wagi (`weights`) są parametrami modelu (`nn.Parameter`), ale ich atrybut `requires_grad` jest ustawiony na `False`, co oznacza, że nie są one uczone przez standardową propagację wsteczną. Wagi są optymalizowane zewnętrznie za pomocą Optuny.
*   **Parametr Temperatury:** Model zawiera parametr *temperatury* (`temperature`, również `nn.Parameter` z `requires_grad=False`), który jest używany do skalowania logitów z modeli składowych przed zastosowaniem funkcji softmax. Pozwala to na kalibrację "ostrości" rozkładu prawdopodobieństw.
*   **Regularyzacja L1 Wag:** Model posiada metodę `l1_regularization()`, która oblicza normę L1 wag. Ten składnik może być dodany do funkcji celu Optuny, aby promować rzadsze wagi (tzn. faworyzować mniejszą liczbę bardziej wpływowych modeli składowych w ensemble). Siła tej regularyzacji jest konfigurowalna.

## Trening i Optymalizacja

Proces treningu modeli indywidualnych oraz modelu ensemble jest starannie zarządzany i monitorowany.

### Trening Indywidualnych Modeli AudioResNet

Modele AudioResNet są trenowane niezależnie dla każdej z jedenastu reprezentacji cech w celu oceny ich indywidualnej skuteczności w rozpoznawaniu emocji. Ten proces jest szczegółowo zaimplementowany w notatniku `src/ResNet_porownanie.ipynb` (głównie w funkcjach `train_model_for_feature` i `run_feature_comparison_experiment`). Kluczowe aspekty treningu:
*   **Funkcja Straty:** `nn.CrossEntropyLoss`.
*   **Optymalizator:** `torch.optim.Adam` (konfigurowany z `LEARNING_RATE` i `WEIGHT_DECAY` z `src/config.py`).
*   **Harmonogram Uczenia (Learning Rate Scheduler):** `torch.optim.lr_scheduler.ReduceLROnPlateau`, który redukuje współczynnik uczenia, gdy strata walidacyjna przestaje maleć (parametry cierpliwości i współczynnika redukcji są konfigurowalne).
*   **Wczesne Zatrzymywanie (Early Stopping):** Zaimplementowano mechanizm wczesnego zatrzymywania (klasa `EarlyStopping` w `src/helpers/early_stopping.py`), który monitoruje stratę na zbiorze walidacyjnym i automatycznie przerywa trening, jeśli przez określoną liczbę epok (`EARLY_STOPPING_PATIENCE` z `src/config.py`) nie następuje poprawa. Zapisywany jest model z najlepszą dotychczasową stratą walidacyjną.
*   **Zapis Wyników:** Dla każdego wytrenowanego modelu zapisywane są: najlepszy stan modelu (`.pt`), historia treningu (straty, dokładności w plikach tekstowych i na wykresach `.png`), raport klasyfikacji (`.csv`), macierz pomyłek (`.png`) oraz plik tekstowy z podsumowaniem hiperparametrów i metryk. Wyniki te trafiają do dedykowanych podkatalogów w `src/feature_comparison_results/` (konfigurowalne jako `FEATURE_RESULTS_DIR`).

### Trening i Optymalizacja Modelu Ensemble

Proces "treningu" modelu ensemble (`WeightedEnsembleModel`) polega głównie na **optymalizacji wag**, z jakimi łączone są predykcje modeli składowych. Realizowane jest to w notatniku `src/ResNet_ensemble.ipynb` przy użyciu klasy `EnsembleModelTrainer` (z `src/helpers/ensemble_trainer.py`).
*   **Automatyczna Konfiguracja:** Skrypty (`auto_configure_ensemble` w notatniku) automatycznie wyszukują najlepsze wytrenowane modele indywidualne (dla cech melspektrogram, MFCC, chroma) oraz odpowiadające im przetworzone pliki cech z katalogów `FEATURE_RESULTS_DIR` i `PROCESSED_FEATURES_DIR`.
*   **Optymalizacja Wag za pomocą Optuny:** Biblioteka **Optuna** jest wykorzystywana do automatycznego przeszukiwania przestrzeni wag w celu znalezienia kombinacji, która maksymalizuje dokładność (lub minimalizuje stratę) modelu ensemble. Proces ten odbywa się z wykorzystaniem **walidacji krzyżowej** (`stratified_kfold_split` na danych treningowo-walidacyjnych, liczba foldów `CV_FOLDS` z `config.py`) w ramach funkcji celu Optuny, aby uniknąć przeuczenia wag do zbioru testowego. Liczba prób Optuny (`OPTUNA_TRIALS`) i limit czasu (`OPTUNA_TIMEOUT`) są konfigurowalne.
*   **Śledzenie Eksperymentów (MLflow):** Przebieg optymalizacji wag za pomocą Optuny, w tym testowane kombinacje wag i uzyskane metryki, jest logowany do **MLflow**. Tworzony jest nowy eksperyment dla każdego uruchomienia optymalizacji.
*   **Finalna Ewaluacja:** Po znalezieniu optymalnych wag, model `WeightedEnsembleModel` jest tworzony z tymi wagami i oceniany na odłożonym zbiorze testowym. Wyniki (metryki, macierz pomyłek) są zapisywane w katalogu `ENSEMBLE_OUTPUT_DIR`.

### Śledzenie Eksperymentów (MLflow)

Poza logowaniem optymalizacji Optuny dla modelu ensemble, MLflow jest wykorzystywany szerzej do śledzenia przebiegu różnych eksperymentów, użytych hiperparametrów, metryk wydajności (np. straty, dokładność, F1-score) oraz wygenerowanych artefaktów (np. wytrenowane modele, wykresy). Domyślnie MLflow zapisuje swoje dane w lokalnym katalogu `mlruns/`. Zapewnia to centralne repozytorium dla wyników, ułatwiając ich porównywanie, analizę i replikację.

## Ewaluacja Modeli

Skuteczność modeli jest oceniana przy użyciu standardowych metryk klasyfikacji, wyliczanych na zbiorze testowym. Kluczowe metryki obejmują:
*   **Accuracy (Dokładność Ogólna):** Procent poprawnie sklasyfikowanych próbek.
*   **F1-score:** Średnia harmoniczna precyzji (precision) i pełności (recall), często wyliczana dla poszczególnych klas emocji oraz uśredniona (np. macro average, weighted average) w celu uwzględnienia potencjalnej nierównowagi klas. W projekcie raporty klasyfikacji zawierają te wartości.
*   **Macierz Pomyłek (Confusion Matrix):** Wizualna reprezentacja wyników klasyfikacji, pokazująca liczbę poprawnych i błędnych predykcji dla każdej pary klas (rzeczywista vs. przewidziana). Generowane są znormalizowane macierze pomyłek i zapisywane jako obrazy.
*   **Raport Klasyfikacji:** Szczegółowy raport zawierający precyzję, pełność i F1-score dla każdej klasy, zapisywany w formacie CSV.

Wyniki dla poszczególnych modeli indywidualnych (na różnych cechach) są porównywane ze sobą (np. za pomocą zbiorczych wykresów dokładności i czasów treningu generowanych w `src/ResNet_porownanie.ipynb`) oraz z wynikami modelu ensemble, aby ocenić, które cechy są najbardziej informatywne i czy technika uczenia zespołowego przynosi poprawę wydajności.

## Struktura Projektu i Główne Skrypty/Notatniki

Projekt został zorganizowany w sposób modularny:
*   `src/config.py`: Globalne stałe konfiguracyjne (ścieżki, parametry modeli, parametry ekstrakcji cech, ustawienia Optuny, MLflow itp.).
*   `src/create_data.py`: Skrypt pomocniczy do pobierania (z Hugging Face) i wstępnego przygotowania zbioru danych nEMO (tworzenie pliku `dataset_info.csv`).
*   `src/helpers/`: Katalog zawierający moduły pomocnicze:
    *   `resnet_model_definition.py`: Definicja modelu `AudioResNet` (bazującego na ResNet18).
    *   `augment_for_all_types.py`: Implementacja systemu augmentacji danych audio (wzorzec fabryki i strategie dla różnych typów cech).
    *   `early_stopping.py`: Implementacja mechanizmu wczesnego zatrzymywania treningu.
    *   `ensemble_model.py`: Definicja modelu `WeightedEnsembleModel` (z ważeniem, temperaturą, regularyzacją L1).
    *   `ensemble_trainer.py`: Klasa `EnsembleModelTrainer` do zarządzania ładowaniem modeli bazowych, optymalizacją wag ensemble (z Optuną i CV) oraz ewaluacją.
    *   `ensemble_dataset.py`: Definicje `EnsembleDatasetIndexed` i `ensemble_collate_fn` do obsługi danych dla modelu ensemble z wielu plików cech.
    *   `data_proccesing.py`: Funkcje do wczytywania i agregowania wyników z plików, np. z eksperymentów porównawczych.
    *   `utils.py`: Różne funkcje pomocnicze, np. `load_pretrained_model`, `evaluate_model` (używana w `ensemble_trainer`), `stratified_kfold_split`.
    *   `visualization.py`: Funkcje do generowania wizualizacji, np. wykresów porównawczych, macierzy pomyłek.
*   `src/ResNet_porownanie.ipynb`: Główny notatnik Jupyter do przeprowadzania eksperymentów porównawczych dla indywidualnych modeli `AudioResNet` na jedenastu różnych reprezentacjach audio. Obejmuje ekstrakcję i buforowanie cech, trening modeli (z augmentacją i wczesnym zatrzymywaniem), ewaluację oraz generowanie zbiorczych wyników i wizualizacji.
*   `src/ResNet_ensemble.ipynb`: Notatnik Jupyter dedykowany budowie, optymalizacji (za pomocą Optuny i `EnsembleModelTrainer`) i ewaluacji modelu ensemble (`WeightedEnsembleModel`).
*   `src/export_scripts/`: Katalog ze skryptami do eksportu modelu ensemble do formatu ONNX oraz testowania modeli ONNX.
    *   `export_to_onnx.py`: Główny skrypt eksportujący model PyTorch (`WeightedEnsembleModel`) do ONNX, wykonujący optymalizację ONNX i weryfikację poprawności konwersji.
    *   `test_onnx_model.py`: Skrypt do testowania inferencji na zapisanym (i zoptymalizowanym) modelu ONNX przy użyciu konkretnego pliku audio (ekstrahuje cechy, przeprowadza inferencję, wizualizuje wyniki).
    *   `test_quantization.py`: Skrypt do przeprowadzania kwantyzacji modelu ONNX (różne strategie), porównywania rozmiarów modeli oraz oceny wpływu kwantyzacji na zgodność predykcji (na losowo generowanych danych wejściowych).
    *   `ensemble_onnx_wrapper.py`: Moduł pomocniczy opakowujący funkcje związane z eksportem, optymalizacją, kwantyzacją i weryfikacją modeli ONNX.
    *   `test_without_audio.py`: Moduł pomocniczy używany w `test_quantization.py`, zawierający funkcje do generowania losowych cech i znajdowania modeli bez potrzeby przetwarzania plików audio.
*   `src/feature_comparison_results/`: Katalog (konfigurowalny jako `FEATURE_RESULTS_DIR`), w którym zapisywane są szczegółowe wyniki (modele `.pt`, metryki, wykresy) dla każdego indywidualnego modelu trenowanego w `ResNet_porownanie.ipynb`.
*   `src/processed_features/`: Katalog (konfigurowalny jako `PROCESSED_FEATURES_DIR`) przechowujący zbuforowane (przetworzone) cechy audio w formacie `.pkl`.
*   `src/ensemble_outputs/`: Katalog (konfigurowalny jako `ENSEMBLE_OUTPUT_DIR`) na wyniki związane z modelem ensemble (zapisane modele, raporty ewaluacyjne, wykresy z optymalizacji Optuny).
*   `src/exported_models/`: Domyślny katalog na wyeksportowane modele ONNX i ich metadane.
*   `src/test_results/`: Domyślny katalog na wyniki testów modeli ONNX (np. wizualizacje z `test_onnx_model.py`).
*   `src/mlruns/`: Katalog tworzony przez MLflow, przechowujący dane eksperymentów (parametry, metryki, artefakty). Lokalizacja może być skonfigurowana w MLflow.
*   `requirements.txt`: Lista zależności projektu (generowana np. przez `uv pip freeze > requirements.txt`).
*   `pyproject.toml`: Plik konfiguracyjny dla narzędzi takich jak `ruff` (linting/formatowanie) i `uv` (zarządzanie zależnościami i środowiskiem wirtualnym).
*   `.ruff.toml`: Dodatkowa konfiguracja dla `ruff` (może być scalona z `pyproject.toml`).

## Eksport Modelu do ONNX

Wytrenowany i zoptymalizowany model ensemble (`WeightedEnsembleModel`) jest eksportowany do formatu **ONNX (Open Neural Network Exchange)**. ONNX to otwarty standard umożliwiający interoperacyjność modeli między różnymi frameworkami uczenia maszynowego i narzędziami. Eksport do ONNX jest kluczowy dla wdrożenia modelu w środowiskach produkcyjnych.

Proces eksportu, zaimplementowany głównie w `src/export_scripts/export_to_onnx.py` przy użyciu funkcji z `src/export_scripts/ensemble_onnx_wrapper.py`, obejmuje:
1.  **Ładowanie Modelu Ensemble:** Wczytanie zapisanego modelu `WeightedEnsembleModel` (pliku `.pt`) wraz z jego wagami i konfiguracją modeli bazowych.
2.  **Generowanie Przykładowych Danych Wejściowych:** Utworzenie krotki tensorów (`dummy_input`) o odpowiednich kształtach dla każdej z cech wejściowych modelu ensemble (melspektrogram, mfcc, chroma). Kształty te są dynamicznie wyliczane na podstawie parametrów ekstrakcji cech.
3.  **Konwersja do ONNX:** Użycie funkcji `torch.onnx.export` do przekształcenia modelu PyTorch do formatu ONNX. Ustawiany jest `opset_version` (np. 17) oraz definiowane są nazwy wejść i wyjść modelu ONNX.
4.  **Optymalizacja Modelu ONNX:** Zastosowanie narzędzi takich jak `onnxoptimizer` (opakowane w `ensemble_onnx_wrapper.py`) w celu redukcji rozmiaru modelu i potencjalnego zwiększenia wydajności inferencji poprzez uproszczenie grafu obliczeniowego.
5.  **Weryfikacja Modelu ONNX:** Porównanie wyjść zoptymalizowanego modelu ONNX z wyjściami oryginalnego modelu PyTorch dla tych samych `dummy_input`, aby upewnić się, że konwersja i optymalizacja przebiegły poprawnie i nie zmieniły znacząco predykcji.
6.  **Kwantyzacja (Opcjonalnie):** Możliwość dalszej optymalizacji poprzez kwantyzację (np. do INT8 lub UINT8, dynamiczna lub statyczna). Skrypt `src/export_scripts/test_quantization.py` pozwala na przeprowadzenie różnych typów kwantyzacji i ocenę ich wpływu na rozmiar modelu oraz zgodność predykcji z modelem niekwantyzowanym.
7.  **Zapis Metadanych:** Zapisanie informacji o modelu ONNX (typy i nazwy cech wejściowych, nazwy klas, oczekiwane kształty tensorów wejściowych i wyjściowych) w pliku JSON, co ułatwia późniejsze wykorzystanie modelu.

Dzięki temu pipeline'owi, wytrenowany model ensemble jest przygotowany do wdrożenia w różnych środowiskach obsługujących ONNX Runtime.

## Technologie i Biblioteki

Projekt wykorzystuje szereg narzędzi i bibliotek programistycznych:
*   **Język programowania:** Python (>=3.9)
*   **Główna biblioteka uczenia maszynowego:** `PyTorch` (budowa i trening modeli neuronowych)
*   **Przetwarzanie i analiza audio:** `librosa` (ekstrakcja cech, analiza sygnału audio)
*   **Biblioteki numeryczne i przetwarzanie danych:** `numpy` (operacje na tablicach), `pandas` (manipulacja danymi tabelarycznymi, np. metadane zbioru, wyniki eksperymentów)
*   **Wizualizacja danych:** `matplotlib`, `seaborn`, `plotly` (generowanie wykresów, wizualizacji cech i wyników)
*   **Zarządzanie eksperymentami ML:** `MLflow` (śledzenie eksperymentów, logowanie parametrów, metryk i artefaktów, np. modeli, wykresów Optuny)
*   **Optymalizacja hiperparametrów:** `Optuna` (optymalizacja wag modelu ensemble z wykorzystaniem CV)
*   **Narzędzia ML/DL:** `scikit-learn` (metryki ewaluacyjne, podział danych, kodowanie etykiet), `joblib` (potencjalnie do przetwarzania równoległego, choć nie jest to główny użytek w obecnej wersji)
*   **Format wymiany modeli i inferencja:** `ONNX`, `ONNX Runtime` (uruchamianie modeli ONNX), `onnxoptimizer` (optymalizacja grafu modeli ONNX), `onnxsim` (uproszczenie modeli ONNX - może być używane zamiennie lub dodatkowo do `onnxoptimizer`)
*   **Obsługa zbiorów danych:** `datasets` (biblioteka Hugging Face do łatwego ładowania i zarządzania zbiorami danych, np. nEMO)
*   **Serializacja obiektów:** `pickle` (buforowanie przetworzonych cech audio)
*   **Obsługa plików i systemu:** `os`, `glob` (wyszukiwanie plików), `json` (zapis/odczyt metadanych), `argparse` (parsowanie argumentów linii komend dla skryptów)
*   **Środowisko pracy:** `Jupyter Notebook` (interaktywne prototypowanie, analiza danych, prowadzenie eksperymentów)
*   **Zarządzanie zależnościami i środowiskiem wirtualnym:** `uv` (szybkie zarządzanie zależnościami na podstawie `pyproject.toml` i `uv.lock`)
*   **Linting/Formatowanie kodu:** `ruff` (skonfigurowane w `pyproject.toml` do zapewnienia spójności i jakości kodu)
*   **System kontroli wersji:** `Git`, `GitHub` (zarządzanie kodem źródłowym, współpraca)

## Przykładowe Komendy / Użycie

Poniżej przedstawiono przykładowe komendy demonstrujące użycie wybranych skryptów w projekcie. Przed uruchomieniem upewnij się, że wszystkie zależności są zainstalowane (np. za pomocą `uv sync` w środowisku wirtualnym `uv`) i że posiadasz odpowiednie pliki modeli (np. wytrenowane w notatnikach `ResNet_porownanie.ipynb` i `ResNet_ensemble.ipynb`).

*   **Eksport modelu ensemble do ONNX (używając najnowszego zapisanego modelu ensemble z `ensemble_outputs/`):**
    ```bash
    python src/export_scripts/export_to_onnx.py
    ```
    Lub ze specyficznym modelem `.pt` i katalogiem wyjściowym dla modeli ONNX:
    ```bash
    python src/export_scripts/export_to_onnx.py --model_path src/ensemble_outputs/ensemble_run_YYYYMMDD_HHMMSS/models/ensemble_model.pt --output_dir src/exported_models/my_onnx_export
    ```

*   **Testowanie zoptymalizowanego modelu ONNX na pliku audio:**
    ```bash
    python src/export_scripts/test_onnx_model.py --model_path src/exported_models/onnx_YYYYMMDD_HHMMSS/ensemble_model_optimized.onnx --audio_path data/test/przykladowy_plik.wav
    ```
    (Zakładając, że plik audio znajduje się w `data/test/`. Skrypt spróbuje znaleźć domyślny plik audio, jeśli nie zostanie podany).

*   **Kwantyzacja modelu ONNX i testowanie wpływu na wydajność i rozmiar:**
    ```bash
    python src/export_scripts/test_quantization.py --model_path src/exported_models/onnx_YYYYMMDD_HHMMSS/ensemble_model_optimized.onnx
    ```
    (Skrypt ten stworzy skwantyzowane modele w podkatalogu `quantized_models` wewnątrz katalogu z oryginalnym modelem ONNX i wygeneruje raporty oraz wizualizacje porównawcze).

## Możliwości Rozbudowy i Przyszłe Kierunki

Architektura projektu została zaprojektowana z myślą o modułowości, co ułatwia jego dalszą rozbudowę i prowadzenie nowych eksperymentów. Możliwe kierunki rozwoju obejmują:
*   **Eksploracja Nowych Architektur Modeli:** Testowanie innych wariantów ResNet (np. ResNet34, ResNet50) lub zupełnie innych architektur CNN (np. EfficientNet, MobileNet), a także modeli opartych o transformery (np. HuBERT, SpeechBrain, Whisper) dostosowanych do zadań SER.
*   **Badanie Dodatkowych Reprezentacji Audio:** Implementacja i ocena nowych, potencjalnie bardziej zaawansowanych typów cech akustycznych lub technik ich automatycznej ekstrakcji.
*   **Rozwój Technik Augmentacji:** Wprowadzenie i ewaluacja nowych, bardziej zaawansowanych strategii augmentacji danych, np. opartych o modele generatywne (GANs, VAEs) lub techniki mieszania próbek (MixUp, CutMix) zaadaptowane do audio.
*   **Zaawansowane Metody Uczenia Zespołowego:** Badanie alternatywnych technik łączenia modeli, takich jak stacking z meta-uczniem, boosting, czy dynamiczny wybór modeli w zespole w zależności od charakterystyki próbki wejściowej.
*   **Dalsza Optymalizacja Hiperparametrów:** Systematyczne strojenie hiperparametrów zarówno modeli indywidualnych (np. architektury, parametrów treningu), jak i parametrów modelu ensemble (np. temperatury, siły regularyzacji L1, wyboru modeli składowych) przy użyciu zaawansowanych strategii HPO (np. Bayesian Optimization, ewolucyjne algorytmy).
*   **Ulepszenie Pipeline'u MLOps:** Rozwój automatyzacji całego cyklu życia modelu ML (trening, ewaluacja, wersjonowanie danych i modeli, wdrożenie) przy użyciu narzędzi takich jak DVC (Data Version Control), Kubeflow, Airflow, GitHub Actions, czy integracja z platformami chmurowymi (AWS SageMaker, Google AI Platform, Azure ML).
*   **Wdrożenie w Czasie Rzeczywistym:** Rozwój rozwiązań umożliwiających inferencję modelu w czasie rzeczywistym, np. w formie mikroserwisu (FastAPI, Flask), aplikacji webowej lub mobilnej, z wykorzystaniem zoptymalizowanych środowisk uruchomieniowych dla ONNX (np. ONNX Runtime z różnymi providerami wykonawczymi - CPU, GPU, TensorRT).
*   **Analiza Interpretowalności Modeli:** Zastosowanie technik XAI (Explainable AI), takich jak SHAP, LIME, czy metody oparte na gradientach (np. Grad-CAM dla audio), w celu lepszego zrozumienia, na które fragmenty sygnału lub cechy model zwraca uwagę podczas podejmowania decyzji.

Projekt stanowi solidną podstawę do dalszych badań nad rozpoznawaniem emocji w mowie i może być rozwijany zarówno w celach naukowych, jak i komercyjnych.
