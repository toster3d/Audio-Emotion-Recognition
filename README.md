# Rozpoznawanie Emocji w Nagraniach Audio

## Wstęp

Celem projektu jest stworzenie modelu rozpoznającego emocje w nagraniach audio. Rozpoznawanie emocji w mowie ma szerokie zastosowanie w interakcji człowiek-komputer, analizie sentymentu i wielu innych dziedzinach.

Projekt wykorzystuje zestaw danych **nEMO**, który zawiera nagrania mowy z przypisanymi etykietami emocji.

## Analiza Danych

Zestaw danych nEMO charakteryzuje się zróżnicowanymi nagraniami mowy. W ramach projektu przeprowadzona została analiza statystyk dotyczących nagrań i rozkładu emocji w zbiorze.

## Przetwarzanie Danych

Dane audio są przetwarzane w celu ekstrakcji cech istotnych dla rozpoznawania emocji. W ramach projektu eksperymentuje się z **jedenastoma różnymi reprezentacjami audio**, w tym:

- **Spektrogramy Mel'a**
- **Spektrogramy Liniowe**
- **MFCC (Mel-frequency cepstral coefficients)**
- **Chromagramy**
- **Spectral Contrast**
- **Zero Crossing Rate (ZCR)**
- **Root Mean Square Energy (RMS)**
- **Tempogram**
- **Tonnetz**
- **Delta MFCC**
- **Delta Tempogram**

Zaimplementowano rozbudowany mechanizm **augmentacji danych**, który wykorzystuje **wzorzec fabryki (factory pattern)** do dynamicznego stosowania specyficznych transformacji w zależności od typu reprezentacji audio. Obejmuje to techniki takie jak dodawanie szumu, maskowanie w dziedzinie czasu i częstotliwości, czy przesunięcie w czasie, dostosowane do charakterystyki poszczególnych cech. Celem augmentacji jest zwiększenie różnorodności zbioru treningowego, co pomaga w zapobieganiu przeuczeniu i poprawie generalizacji modelu.

Dane są dzielone na zbiory treningowe, walidacyjne i testowe. Przetworzone cechy audio mogą być buforowane na dysku w celu przyspieszenia kolejnych eksperymentów.

## Wybór i Implementacja Modelu

Projekt wykorzystuje konwolucyjne sieci neuronowe (CNN) do klasyfikacji emocji na podstawie przetworzonych cech audio. Głównym modelem jest **AudioResNet**, bazujący na architekturze **ResNet18**, zaimplementowany w bibliotece **PyTorch**. Modele są trenowane indywidualnie dla każdej z jedenastu reprezentacji audio.

Dodatkowo, w projekcie rozwijany jest **model ensemble** składający się z trzech modeli AudioResNet wytrenowanych na reprezentacjach **melspektrogram, MFCC i chroma**. Model ensemble łączy predykcje modeli indywidualnych w celu uzyskania lepszych wyników.

## Trening i Optymalizacja

Proces treningu modeli obejmuje:

- Trening **indywidualnych modeli AudioResNet** dla każdej z jedenastu reprezentacji audio w celu oceny ich skuteczności.
- Trening **modelu ensemble** opartego na wybranych reprezentacjach (melspektrogram, MFCC, chroma).
- **Optymalizacja wag modeli składowych** w modelu ensemble przy użyciu biblioteki **Optuna**. Proces ten ma na celu znalezienie najlepszej kombinacji wag, która maksymalizuje dokładność klasyfikacji.
- Śledzenie przebiegu eksperymentów, parametrów i metryk za pomocą narzędzia **MLflow**. MLflow służy do monitorowania procesu treningu i optymalizacji, logowania wyników oraz zarządzania artefaktami, takimi jak wytrenowane modele i wykresy.
- Zastosowanie technik **regularyzacji** (np. Dropout) w celu zapobiegania przeuczeniu.
- Wykorzystanie mechanizmu **wczesnego zatrzymywania** (Early Stopping) monitorującego straty walidacyjne w celu przerwania treningu w odpowiednim momencie.
- Planowana jest dalsza **optymalizacja hiperparametrów** modeli indywidualnych i ensemble oraz eksploracja **technik uczenia zespołowego**.

## Ewaluacja Modelu

Model jest ewaluowany przy użyciu standardowych metryk oceny dla zadań klasyfikacji, takich jak **accuracy**, **F1-score** oraz **macierz pomyłek (confusion matrix)**. Analiza błędów i porównanie wyników dla różnych konfiguracji modelu są kluczowe dla oceny jego skuteczności.

## Struktura Projektu i Przebieg Eksperymentów

Projekt jest zorganizowany w sposób umożliwiający przeprowadzanie eksperymentów z różnymi reprezentacjami audio i modelami. Kluczowym elementem jest notatnik **`src/ResNet_porownanie.ipynb`**, który stanowi kompleksowe środowisko do testowania skuteczności modelu ResNet18 na jedenastu różnych reprezentacjach dźwięku. Architektura notatnika obejmuje:

1.  **Ładowanie i Przygotowanie Danych:** Proces wczytywania zestawu danych nEMO.
2.  **Ekstrakcja Cech Audio:** Wyodrębnianie jedenastu zdefiniowanych reprezentacji (spektrogramy Mel'a, MFCC, chroma, itp.) z surowych nagrań audio. W celu przyspieszenia eksperymentów, **przetworzone cechy audio są buforowane na dysku (caching)**, co pozwala uniknąć wielokrotnego przetwarzania tych samych danych przy ponownych uruchomieniach notatnika.
3.  **Trening Indywidualnych Modeli:** Trening osobnego modelu AudioResNet (ResNet18) dla każdej z jedenastu wyekstrahowanych reprezentacji audio.
4.  **Ewaluacja Wyników:** Ocena wydajności każdego z indywidualnie trenowanych modeli przy użyciu standardowych metryk.
5.  **Wizualizacja Wyników:** Generowanie wykresów porównujących wyniki modeli trenowanych na różnych reprezentacjach, co ułatwia analizę i wybór najskuteczniejszych cech.

Równolegle rozwijany jest **model ensemble** w notatniku **`src/ResNet_ensemble.ipynb`**. Notatnik ten skupia się na:

1.  **Konfiguracji Modelu Ensemble:** Automatyczne wyszukiwanie i ładowanie indywidualnych modeli ResNet (obecnie trenowanych na melspektrogramach, MFCC i chromie) oraz odpowiadających im przetworzonych danych.
2.  **Optymalizacji Wag Ensemble:** Wykorzystanie biblioteki Optuna do znalezienia optymalnych wag, z jakimi powinny być łączone predykcje modeli składowych ensemble.
3.  **Ewaluacji Modelu Ensemble:** Ocena wydajności modelu ensemble na zbiorze testowym.
4.  **Analizie Błędów:** Identyfikacja przypadków, w których model ensemble popełnia błędy klasyfikacji.

Proces wdrożenia modelu do środowisk produkcyjnych jest realizowany poprzez **eksport modelu ensemble do formatu ONNX**, opisanego w dedykowanej sekcji.

## Wnioski i Przyszłe Kierunki

Projekt jest w fazie rozwoju. Dotychczasowe prace skupiają się na stworzeniu solidnej podstawy do ekstrakcji cech i implementacji modeli ResNet w PyTorch, optymalizacji modelu ensemble oraz jego eksportu do formatu ONNX. Przyszłe kroki obejmują:

- Dalsze eksperymentowanie z różnymi reprezentacjami audio i architekturami modeli.
- **Testowanie alternatywnych metod augmentacji danych** w celu dalszego zwiększenia odporności modelu.
- Dalszą optymalizację hiperparametrów modeli indywidualnych i ensemble.
- Implementację i ewaluację dodatkowych technik uczenia zespołowego.
- **Eksploracja innych sposobów łączenia różnych reprezentacji audio**, na przykład poprzez połączenie kilku typów reprezentacji jednocześnie (np. MFCC, chroma, Tonnetz) w **wielokanałowy obraz** podawany na wejście sieci konwolucyjnej.
- Dalszą analizę wyników i błędów w celu ulepszenia modelu.
- Rozwinięcie procesów MLOps, w tym automatyzację treningu, ewaluacji i eksportu.

## Eksport Modelu do ONNX

Model ensemble może zostać wyeksportowany do formatu **ONNX (Open Neural Network Exchange)**. Eksport do ONNX umożliwia wdrożenie modelu w różnych środowiskach i platformach, w tym na urządzeniach mobilnych czy w przeglądarkach internetowych, dzięki zoptymalizowanym środowiskom uruchomieniowym (np. ONNX Runtime).

Proces eksportu obejmuje:

1.  **Konwersję** modelu PyTorch do formatu ONNX.
2.  **Optymalizację** wyeksportowanego modelu ONNX przy użyciu dedykowanych narzędzi w celu poprawy wydajności inferencji.
3.  Opcjonalną **kwantyzację** modelu w celu redukcji jego rozmiaru i dalszego przyspieszenia inferencji, co jest szczególnie przydatne w środowiskach z ograniczonymi zasobami.
4.  **Weryfikację** wyeksportowanego modelu ONNX poprzez porównanie jego wyjść z wyjściami oryginalnego modelu PyTorch w celu zapewnienia zgodności.

Eksportu można dokonać przy użyciu dedykowanych skryptów (`export_to_onnx.py`) lub funkcji pomocniczych w notatnikach Jupyter (`export_onnx_notebook_helper.py`).

## Technologie

- **Język programowania:** Python
- **Biblioteki do przetwarzania audio:** `librosa`, `tensorflow-io` (do ładowania danych), `pyAudioAnalysis`
- **Biblioteka uczenia maszynowego:** `PyTorch`
- **Format wymiany modeli ML:** `ONNX`
- **Środowisko uruchomieniowe ONNX:** `ONNX Runtime`
- **Narzędzia do wizualizacji:** `matplotlib`, `seaborn`, `librosa.display`
- **Narzędzia do zarządzania eksperymentami:** `MLflow`, `TensorBoard`
- **Narzędzia do optymalizacji hiperparametrów:** `Optuna`
- **Środowisko pracy:** Jupyter Notebook, Google Colab z GPU
- **Narzędzia do wersjonowania kodu:** Git, GitHub

## Architektura projektu

Projekt został zaprojektowany w sposób modularny, zgodnie z najlepszymi praktykami inżynierii ML. Każdy etap przetwarzania danych, treningu, ewaluacji, ensemble, eksportu i testowania modeli jest wydzielony do osobnych modułów i skryptów. Pipeline projektu można przedstawić następująco:

1. **Ekstrakcja cech audio** (różne reprezentacje, caching wyników)
2. **Augmentacja danych** (modularna, oparta o wzorzec fabryki)
3. **Trening modeli indywidualnych** (ResNet18 dla każdej reprezentacji)
4. **Trening i optymalizacja modelu ensemble** (łączenie predykcji, automatyczna optymalizacja wag)
5. **Ewaluacja i analiza wyników** (automatyczne zbieranie, wizualizacja, analiza błędów)
6. **Eksport do ONNX** (optymalizacja, kwantyzacja, testy zgodności)
7. **Testowanie i wdrożenie** (skrypty do testowania modeli ONNX, porównania, wizualizacje)

Każdy z tych etapów jest zaimplementowany w dedykowanych plikach, co ułatwia rozwój i utrzymanie projektu.

## Zaawansowane mechanizmy augmentacji

Projekt wykorzystuje rozbudowany, modularny system augmentacji danych audio, oparty o **wzorzec fabryki (factory pattern)**. Dla każdej reprezentacji cech audio (np. melspektrogram, MFCC, chroma, tempogram, spectral contrast, ZCR, RMS) stosowane są dedykowane strategie augmentacji, takie jak:
- dodawanie szumu,
- maskowanie w dziedzinie czasu i częstotliwości,
- przesunięcia w czasie,
- maskowanie harmoniczne,
- augmentacje specyficzne dla cech jednowymiarowych.

Fabryka augmentacji umożliwia łatwe dodawanie nowych strategii i dynamiczne dobieranie transformacji do typu cechy. To rozwiązanie znacząco zwiększa elastyczność i skalowalność pipeline'u.

## Automatyzacja eksperymentów i zarządzanie wynikami

Projekt implementuje szereg narzędzi automatyzujących eksperymenty:
- **Optuna** – automatyczna optymalizacja wag modelu ensemble, pozwalająca na znalezienie najlepszych kombinacji bez ręcznego strojenia.
- **MLflow** – śledzenie eksperymentów, hiperparametrów, metryk, artefaktów (modele, wykresy), co umożliwia replikowalność i analizę postępów.
- **Wczesne zatrzymywanie (EarlyStopping)** – własna implementacja monitorująca straty walidacyjne i automatycznie przerywająca trening, gdy model przestaje się poprawiać.
- **Automatyczne zbieranie i agregacja wyników** – funkcje do odczytu i analizy wyników z wielu eksperymentów, zarówno na poziomie ogólnym, jak i dla poszczególnych emocji.
- **Wizualizacja wyników** – generowanie wykresów porównawczych, macierzy pomyłek, analiz błędów.

## Obsługa i testowanie modeli ONNX

Projekt zawiera kompletny pipeline do eksportu, optymalizacji i testowania modeli ONNX:
- **Eksport modelu ensemble do ONNX** (skrypty i klasy pomocnicze)
- **Optymalizacja i kwantyzacja modeli ONNX** (redukcja rozmiaru, przyspieszenie inferencji)
- **Automatyczne testy zgodności** – porównanie wyników modeli PyTorch i ONNX, weryfikacja poprawności eksportu
- **Wizualizacja cech i wyników** – narzędzia do wizualizacji wejść, predykcji i porównań modeli
- **Przykładowe skrypty** do testowania modeli ONNX na nowych danych, porównywania modeli po kwantyzacji, itp.

Dzięki temu pipeline'owi model może być łatwo wdrożony w środowiskach produkcyjnych, na urządzeniach mobilnych, w chmurze czy na edge.

## Przykładowe komendy i skrypty

Poniżej przykłady użycia wybranych skryptów:

- Eksport modelu ensemble do ONNX:
  ```bash
  python src/export_scripts/export_to_onnx.py --model_path <ścieżka_do_modelu> --output_dir <ścieżka_wyjściowa>
  ```
- Testowanie modelu ONNX na pliku audio:
  ```bash
  python src/export_scripts/test_onnx_model.py --model_path <ścieżka_do_onnx> --audio_path <ścieżka_do_audio>
  ```
- Kwantyzacja modelu ONNX:
  ```bash
  python src/export_scripts/test_quantization.py --model_path <ścieżka_do_onnx> --output_dir <ścieżka_wyjściowa>
  ```
- Porównanie wyników modeli oryginalnych i po kwantyzacji:
  ```bash
  python src/export_scripts/test_quantization.py --compare --model_path <ścieżka_do_onnx> --output_dir <ścieżka_wyjściowa>
  ```

## Możliwości rozbudowy

Architektura projektu pozwala na łatwe rozszerzanie i modyfikowanie pipeline'u:
- Dodawanie nowych typów cech audio i strategii augmentacji (wystarczy dodać nową klasę i zarejestrować ją w fabryce augmentacji)
- Implementacja nowych architektur modeli (np. inne warianty ResNet, modele 1D, modele hybrydowe)
- Rozbudowa modelu ensemble o kolejne modele bazowe lub inne sposoby łączenia predykcji
- Eksperymentowanie z alternatywnymi metodami augmentacji i łączenia cech (np. wielokanałowe wejścia do sieci konwolucyjnej)
- Integracja z innymi narzędziami MLOps (np. DVC, Neptune)
- Automatyzacja całego pipeline'u (np. przy użyciu Makefile, CI/CD)

Projekt jest gotowy do dalszego rozwoju zarówno w kierunku badań naukowych, jak i wdrożeń produkcyjnych.
