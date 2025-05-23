# Rozpoznawanie Emocji w Nagraniach Audio

## Spis treści
- [Wstęp](#wstęp)
- [Zbiór danych](#zbiór-danych-nemo)
- [Metodologia](#metodologia)
- [Architektura modeli](#architektura-modeli)
- [Ewaluacja i analiza](#ewaluacja-i-analiza)
- [Wdrożenie](#wdrożenie)
- [Technologie](#technologie-i-biblioteki)
- [Rozwój projektu](#możliwości-rozbudowy)

## Wstęp

Projekt koncentruje się na opracowaniu zaawansowanego systemu automatycznego rozpoznawania emocji w mowie (Speech Emotion Recognition - SER) w języku polskim. Systemy SER stanowią kluczowy element nowoczesnych technologii Human-Computer Interaction (HCI) i znajdują szerokie zastosowanie w różnych dziedzinach.

**Zastosowania praktyczne SER:**
- **Centra obsługi klienta:** Automatyczna analiza zadowolenia klientów i eskalacja problemowych rozmów
- **Telemedycyna:** Wspomaganie diagnostyki zaburzeń psychicznych i monitorowania stanu emocjonalnego pacjentów
- **Edukacja:** Personalizacja procesu nauczania w oparciu o reakcje emocjonalne uczniów
- **Systemy smart home:** Adaptacja środowiska do aktualnego stanu emocjonalnego użytkowników
- **Przemysł rozrywkowy:** Tworzenie bardziej immersyjnych doświadczeń w grach i aplikacjach VR/AR

**Cel projektu:** Zbadanie wpływu różnorodnych reprezentacji akustycznych sygnału mowy oraz zastosowanie technik uczenia zespołowego na skuteczność rozpoznawania emocji przy użyciu głębokich konwolucyjnych sieci neuronowych. Projekt szczególnie koncentruje się na języku polskim, który jest znacznie mniej reprezentowany w literaturze naukowej niż język angielski.

## Zbiór Danych: nEMO

Wykorzystano polskojęzyczny zbiór **nEMO** (National Emotional Speech Corpus) z Centrum Sztucznej Inteligencji UAM w Poznaniu, który stanowi jeden z najważniejszych zasobów dla badań nad rozpoznawaniem emocji w języku polskim.

**Charakterystyka zbioru:**
- **Rozmiar:** 4480 nagrań treningowych o zróżnicowanej długości
- **Zakres czasowy:** 0.87-5.66 sekundy (średnio ~2.5 sekundy)
- **Emocje:** 6 klas podstawowych - złość, strach, szczęście, smutek, zaskoczenie, stan neutralny
- **Różnorodność mówców:** Wieloośrodkowy charakter zbierania danych zapewniający reprezentatywność
- **Metadane:** Kompleksowe informacje obejmujące ID pliku, ścieżkę, etykietę emocji, transkrypcję (surową i znormalizowaną), dane demograficzne mówcy

**Unikalne cechy zbioru nEMO:**
- Naturalne nagrania w języku polskim z kontrolowaną jakością akustyczną
- Zbalansowany rozkład emocji minimalizujący bias klasowy
- Wysokiej jakości adnotacje wykonane przez ekspertów
- Kompatybilność ze standardowymi frameworkami ML/DL

**Źródło:** [Hugging Face - nEMO](https://huggingface.co/datasets/amu-cai/nEMO)

## Metodologia

### Przetwarzanie Danych Audio

**Standaryzacja długości nagrań:**
Wszystkie nagrania są zunifikowane do stałej długości 3 sekund poprzez przycinanie dłuższych plików lub dopełnianie krótszych zerami. Ten proces zapewnia spójność wymiarów dla przetwarzania wsadowego i umożliwia efektywne uczenie modeli CNN.

**Ekstrakcja cech akustycznych:**
Projekt implementuje kompleksowy pipeline ekstrakcji 13 różnych reprezentacji akustycznych, które łącznie charakteryzują odmienne aspekty sygnału mowy istotne dla rozpoznawania emocji:

**Cechy spektralne:**
1. **Spektrogramy Mel'a** - reprezentacja energii w skali percepcyjnej Mel, lepiej oddającej ludzką percepcję wysokości dźwięku
2. **Spektrogramy liniowe** - klasyczna reprezentacja STFT z liniowym rozkładem częstotliwości
3. **MFCC** - współczynniki cepstralne w skali Mel, standardowe w rozpoznawaniu mowy
4. **CQT** - transformacja o stałym Q z logarytmicznym rozmieszczeniem pasm częstotliwości

**Cechy harmoniczne i tonalne:**
5. **Chromagramy** - rozkład energii w klasach chromatycznych, istotne dla analizy tonalnej
6. **Tonnetz** - sześciowymiarowa reprezentacja centrów tonalnych i relacji harmonicznych
7. **HPSS** - separacja składowych harmonicznych i perkusyjnych sygnału

**Cechy temporalne i energetyczne:**
8. **Zero Crossing Rate (ZCR)** - częstotliwość przejść przez zero, związana z charakterem dźwięku
9. **RMS Energy** - energia skuteczna sygnału w oknach czasowych
10. **Kontrast spektralny** - różnice energii między szczytami a dolinami w pasmach widma

**Cechy dynamiczne:**
11. **Tempogram** - reprezentacja zmian tempa i rytmu w czasie
12. **Delta MFCC** - pochodne czasowe współczynników MFCC
13. **Delta Tempogram** - pochodne czasowe tempogramu

**Zaawansowany system buforowania:**
Przetworzone cechy są serializowane i cache'owane na dysku w formacie pickle z unikalnymi identyfikatorami opartymi na parametrach ekstrakcji. System ten drastycznie redukuje czas iteracyjnych eksperymentów i zapewnia reprodukowalność wyników.

**Modularny system augmentacji:**
Implementacja wykorzystuje wzorzec projektowy fabryki do zastosowania dedykowanych technik augmentacji dla każdego typu cechy:
- **Augmentacje spektralne:** Maskowanie częstotliwościowo-czasowe, dodawanie szumu gaussowskiego
- **Augmentacje temporalne:** Przesunięcia czasowe, rozciąganie/kompresja
- **Augmentacje harmoniczne:** Perturbacje tonalne dla cech chromatycznych
- **Augmentacje energetyczne:** Modulacja amplitudy dla cech ZCR i RMS

### Przygotowanie Danych

**Stratyfikowany podział danych:**
- 64% trening, 16% walidacja, 20% test z zachowaniem proporcji klas emocji
- Dodatkowa 5-fold walidacja krzyżowa dla optymalizacji hiperparametrów ensemble

**Wielopoziomowa normalizacja:**
- Normalizacja per próbka podczas ekstrakcji cech
- Globalna standaryzacja z-score dla całego zbioru danych
- Adaptacyjna normalizacja dla różnych typów reprezentacji

## Architektura Modeli

### Modele Indywidualne

**AudioResNet - Architektura bazowa**

Model AudioResNet stanowi kluczową adaptację architektury ResNet18 do przetwarzania danych audio. Głębokie sieci rezydualne (ResNet) są szczególnie efektywne w zadaniach klasyfikacji dzięki mechanizmowi skip connections, który rozwiązuje problem zaniku gradientu w głębokich sieciach.

**Modyfikacje architektoniczne:**
- **Warstwa wejściowa:** Adaptacja conv1 do jednokanałowych reprezentacji audio (zamiast RGB)
- **Bloki rezydualne:** Wykorzystanie oryginalnych bloków ResNet18 z BatchNorm i ReLU
- **Warstwa klasyfikacyjna:** Zastąpienie FC layer dedykowaną warstwą z Dropout i mapowaniem na 6 klas emocji
- **Inicjalizacja wag:** Kaiming dla konwolucji, Xavier dla warstw liniowych

**Model VGG16 dla Mel-spektrogramów**

Implementacja VGG16 jako alternatywy dla AudioResNet, szczególnie zoptymalizowana pod mel-spektrogramy:
- **Architektura sekwencyjna:** Stopniowe zwiększanie głębokości z filtrami 3x3
- **Adaptacja kanałów:** Modyfikacja do jednokanałowych danych audio
- **Regularyzacja:** Dropout i BatchNorm dla lepszej generalizacji

**Model CNN TensorFlow**

Lekka architektura CNN w TensorFlow/Keras z zaawansowaną augmentacją:
- **Trzy bloki konwolucyjne:** Progresywne zwiększanie złożoności (32-64-128 filtrów)
- **Global Average Pooling:** Redukcja parametrów i prevention overfittingu
- **Rozszerzona augmentacja:** Modyfikacje sygnału audio przed ekstrakcją cech
- **Wyniki:** 46.8% dokładności testowej przy różnorodnych technikach augmentacji

**Model CNN PyTorch (1D)**

Jednowymiarowa architektura konwolucyjna dla sekwencyjnego przetwarzania audio:
- **Cztery bloki Conv1D:** Hierarchiczna ekstrakcja cech (32-64-128-256 filtrów)
- **BatchNorm1d:** Stabilizacja treningu w każdym bloku
- **Global Average Pooling 1D:** Agregacja cech temporalnych
- **Wysokie wyniki:** 82.76% dokładności testowej, demonstrując skuteczność konwolucji 1D

### Model Ensemble

**WeightedEnsembleModel - Uczenie zespołowe**

Zaawansowany model ensemble łączący predykcje trzech najskuteczniejszych modeli indywidualnych (mel-spektrogram, MFCC, chroma):

**Mechanizm ważenia:**
- **Uczne wagi:** Parametry nn.Parameter optymalizowane przez Optuna
- **Regularyzacja L1:** Promowanie rzadkich wag dla lepszej interpretowalności
- **Walidacja krzyżowa:** 5-fold CV dla uniknięcia overfittingu

**Kalibracja prawdopodobieństw:**
- **Parametr temperatury:** Skalowanie logitów przed softmax
- **Adaptacyjna ostrość:** Dynamiczne dostrajanie rozkładu prawdopodobieństw

### Model Równoległy Conv1D + RNN

**Architektura hybrydowa dla ZCR i RMS Energy**

Innowacyjne podejście łączące mocne strony konwolucji i sieci rekurencyjnych:

**Przygotowanie danych:**
- **Ekstrakcja ramkowa:** ZCR i RMS Energy z oknem 25ms i nakładaniem 10ms
- **Normalizacja sekwencyjna:** Standaryzacja z dopełnianiem do stałej długości
- **Representacja temporalna:** Zachowanie informacji o dynamice czasowej

**Ścieżka Conv1D:**
- **Lokalne wzorce:** Detekcja krótkoterminowych zmian w ZCR i energii
- **Hierarchiczna ekstrakcja:** Wielopoziomowe filtry konwolucyjne
- **MaxPooling:** Redukcja wymiarowości z zachowaniem istotnych cech

**Ścieżka RNN (BiLSTM):**
- **Kontekst dwukierunkowy:** Analiza zależności przyszłość-przeszłość
- **Długoterminowa pamięć:** Modelowanie ewolucji emocji w czasie
- **Dropout:** Regularyzacja między warstwami LSTM

**Fuzja informacji:**
- **Konkatenacja:** Łączenie reprezentacji z obu ścieżek
- **Dense layers:** Nieliniowe mapowanie na przestrzeń decyzyjną
- **Softmax:** Ostateczna klasyfikacja na 6 klas emocji

## Ewaluacja i Analiza

### Kompleksowe Metryki Ewaluacyjne

**Metryki podstawowe:**
- **Accuracy:** Globalny procent poprawnych klasyfikacji z uwzględnieniem nierównowagi klas
- **F1-score:** Harmoniczna średnia precyzji i czułości, kluczowa dla niezbalansowanych zbiorów
- **Macierz pomyłek:** Szczegółowa analiza błędów klasyfikacji między parami emocji
- **Raport klasyfikacji:** Per-class precision, recall, F1-score z macro i weighted averaging

**Analiza błędów:**
- **Confusing pairs:** Identyfikacja najczęściej mylonych par emocji
- **Class-specific performance:** Analiza trudności klasyfikacyjnych poszczególnych emocji
- **Error patterns:** Systematyczne wzorce błędów w predykcjach

### Analiza Podatności na Ataki Adwersalne

**Fast Gradient Sign Method (FGSM):**
Kompleksowa analiza odporności modelu AudioResNet na perturbacje adwersalne:

**Metodologia ataków:**
- **Gradient-based perturbations:** Modyfikacje w kierunku maksymalizacji straty
- **Epsilon scaling:** Testowanie różnych intensywności ataków (ε ∈ [0.001, 0.1])
- **Audio reconstruction:** Griffin-Lim dla konwersji spektrogramów na sygnał audio

**Analiza rezultatów:**
- **Degradacja wydajności:** Kwantyfikacja spadku accuracy w funkcji siły ataku
- **Vulnerability patterns:** Identyfikacja najbardziej podatnych klas emocji
- **Perturbation visualization:** Wizualizacja zmian w spektrogramach

**Implikacje bezpieczeństwa:**
Wyniki demonstrują kruchość modeli głębokiego uczenia na subtelne modyfikacje, podkreślając potrzebę technik adversarial training w aplikacjach krytycznych.

### Analiza XAI (Explainable AI)

**Metody wyjaśnialności:**

**LIME (Local Interpretable Model-agnostic Explanations):**
- **Local surrogate models:** Aproksymacja lokalnych decyzji modelu
- **Perturbation-based:** Analiza wpływu modyfikacji na predykcje
- **Feature attribution:** Identyfikacja kluczowych regionów spektrogramu

**LRP (Layer-wise Relevance Propagation):**
- **Backward relevance:** Propagacja istotności od wyjścia do wejścia
- **Conservation principle:** Zachowanie całkowitej relevance przez warstwy
- **Pixel-level attribution:** Szczegółowe mapowanie istotności

**GradCAM:**
- **Gradient-weighted activation:** Wykorzystanie gradientów ostatniej warstwy konwolucyjnej
- **Class-specific heatmaps:** Wizualizacja regionów istotnych dla każdej klasy
- **High-resolution insights:** Precyzyjne wskazanie obszarów decyzyjnych

**Smooth Saliency Maps:**
- **Noise-reduced attribution:** Wygładzanie map istotności
- **Robust explanations:** Redukcja artefaktów w wizualizacjach
- **Consistent patterns:** Stabilne wyjaśnienia między podobnymi próbkami

**Objektywna weryfikacja:**
Ilościowa ocena skuteczności metod XAI poprzez perturbacyjne testy - maskowanie obszarów wskazanych jako istotne i pomiar wpływu na predykcje modelu.

## Wdrożenie

### Eksport do ONNX

**Kompleksowy pipeline eksportu:**

**Przygotowanie modelu:**
- **Model unification:** Łączenie modeli ensemble w jeden graf obliczeniowy
- **Weight freezing:** Zamrożenie parametrów dla deterministycznych predykcji
- **Input standardization:** Unifikacja formatów wejściowych

**Proces konwersji:**
- **PyTorch to ONNX:** Wykorzystanie torch.onnx.export z opset 17
- **Dynamic shapes:** Obsługa zmiennych rozmiarów batch
- **Metadata preservation:** Zachowanie informacji o klasach i preprocessing

**Optymalizacja ONNX:**
- **Graph optimization:** Eliminacja redundantnych operacji
- **Operator fusion:** Łączenie kompatybilnych operatorów
- **Memory optimization:** Redukcja footprint pamięciowego

**Kwantyzacja zaawansowana:**
- **INT8/UINT8 quantization:** Dramatyczna redukcja rozmiaru modelu
- **Dynamic vs static:** Wybór strategii kwantyzacji w zależności od deployment
- **Accuracy preservation:** Monitoring degradacji wydajności po kwantyzacji

**Weryfikacja końcowa:**
- **Numerical consistency:** Porównanie output'ów PyTorch vs ONNX
- **Performance benchmarking:** Pomiary latency i throughput
- **Cross-platform testing:** Walidacja na różnych środowiskach uruchomieniowych

**Korzyści produkcyjne:**
- **Framework agnostic:** Możliwość deployment'u niezależnie od frameworka treningu
- **Hardware optimization:** Akceleracja na CPU, GPU, NPU, TPU
- **Cloud integration:** Łatwa integracja z platformami chmurowymi (Azure ML, AWS SageMaker)

## Technologie i Biblioteki

**Core ML/DL Framework:**
- **PyTorch:** Główny framework dla budowy, treningu i ewaluacji modeli neuronowych, wybrany ze względu na flexibilność w prototypowaniu i silne wsparcie dla research
- **TensorFlow/Keras:** Alternatywna implementacja dla porównania architektur i technik augmentacji

**Audio Processing & Feature Extraction:**
- **librosa:** Comprehensive library dla ekstrakcji cech akustycznych, analiza spektralna, transformacje audio
- **numpy:** Operacje numeryczne na macierzach, manipulacja tensorów audio
- **scipy:** Dodatkowe funkcje przetwarzania sygnałów

**Data Management & Analysis:**
- **pandas:** Manipulacja metadanych, agregacja wyników eksperymentów, analiza statystyczna
- **datasets (Hugging Face):** Streamlined loading i preprocessing zbioru nEMO
- **pickle:** Persistent caching przetworzonych features dla szybkich iteracji

**Visualization & Analysis:**
- **matplotlib:** Podstawowe plotting dla wykresów treningu, macierzy pomyłek
- **seaborn:** Statystyczne visualizations, heatmapy, distribution plots
- **plotly:** Interaktywne wykresy dla deep analysis i presentation

**ML Ops & Experiment Management:**
- **MLflow:** Comprehensive experiment tracking, parameter logging, model versioning, artifact management
- **Optuna:** Bayesian optimization dla hyperparameter tuning, szczególnie wag ensemble
- **scikit-learn:** Evaluation metrics, data splitting, preprocessing utilities

**Model Deployment & Optimization:**
- **ONNX:** Cross-platform model format dla production deployment
- **ONNX Runtime:** High-performance inference engine z acceleration providers
- **onnxoptimizer/onnxsim:** Graph optimization i simplification tools

**Development Environment:**
- **Jupyter Notebook:** Interactive development, data exploration, experiment prototyping
- **uv:** Modern Python dependency management z lock files i virtual environments
- **ruff:** Fast linting i code formatting dla code quality

**Version Control & Collaboration:**
- **Git/GitHub:** Source code management, collaboration, CI/CD integration

## Możliwości Rozbudowy

**Zaawansowane Architektury:**
- **Transformer-based models:** Adaptacja BERT/GPT dla audio (HuBERT, Wav2Vec2, Whisper embeddings)
- **Multi-modal approaches:** Fuzja audio z tekstem (transkrypcje) i visual features (facial expressions)
- **Attention mechanisms:** Self-attention dla temporal modeling w długich nagraniach
- **Neural Architecture Search:** Automatyczne projektowanie architektur specyficznych dla SER

**Feature Engineering Evolution:**
- **Learned representations:** End-to-end uczenie cech zamiast hand-crafted features
- **Cross-lingual features:** Transfer learning między językami dla low-resource scenarios
- **Emotional prosody modeling:** Specjalistyczne cechy dla intonacji i rytmu emocjonalnego
- **Multi-resolution analysis:** Hierarchiczna analiza w różnych skalach czasowych

**Advanced Ensemble Techniques:**
- **Dynamic ensemble selection:** Adaptacyjny wybór modeli w zależności od charakterystyki input
- **Meta-learning approaches:** Learning to combine models na poziomie meta-uczenia
- **Cascade ensembles:** Hierarchiczne podejmowanie decyzji z early stopping
- **Uncertainty quantification:** Bayesian ensembles z estimation niepewności predykcji

**MLOps & Production Readiness:**
- **Continuous learning:** Online adaptation modeli do nowych danych
- **A/B testing framework:** Systematic comparison różnych wersji modeli w production
- **Model monitoring:** Real-time tracking performance degradation i data drift
- **Automated retraining:** Trigger-based model updates przy degradacji metryki

**Real-time Applications:**
- **Streaming inference:** Real-time emotion recognition z minimal latency
- **Edge deployment:** Optymalizacja dla mobile i IoT devices
- **Microservices architecture:** Scalable deployment z Docker/Kubernetes
- **WebRTC integration:** Browser-based real-time emotion analysis

**Extended XAI Research:**
- **Causal analysis:** Understanding causal relationships w emotional speech
- **Counterfactual explanations:** "What-if" analysis dla decision boundaries
- **Human-AI collaboration:** Interactive explanation systems z user feedback
- **Bias detection:** Systematic analysis demographic i cultural biases

Projekt stanowi solid foundation dla cutting-edge research w Speech Emotion Recognition i oferuje multiple pathways dla academic i commercial development w obszarze affective computing. 