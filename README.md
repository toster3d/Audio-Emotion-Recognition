**1. Wstęp**
- Cel projektu: Stworzenie modelu rozpoznającego emocje w nagraniach audio

- Znaczenie rozpoznawania emocji w mowie

- Krótki opis zestawu danych nEMO

**2. Analiza danych**
- Charakterystyka zestawu danych nEMO

- Statystyki dotyczące nagrań i emocji

- Wyzwania związane z danymi (np. nierównowaga klas, jakość nagrań)

**3. Przetwarzanie danych**
- Ekstrakcja różnorodnych reprezentacji audio:

 -- Spektrogramy Mel'a

 -- MFCC (Mel-frequency cepstral coefficients)

 -- Spektrogramy logarytmiczne

 -- Chromagramy

- Normalizacja i augmentacja danych

- Podział na zbiory treningowe, walidacyjne i testowe

**4. Wybór i implementacja modelu**
- Implementacja modelu CNN z wykorzystaniem TensorFlow

- Eksperymentowanie z różnymi architekturami CNN dla różnych reprezentacji audio

- Implementacja modelu ensemble łączącego wyniki z różnych reprezentacji

**5. Trening i optymalizacja**
- Proces treningu modelu z wykorzystaniem różnych reprezentacji audio

- Techniki regularyzacji i zapobiegania przeuczeniu

- Optymalizacja hiperparametrów

- Implementacja technik uczenia zespołowego (ensemble learning)

**6. Ewaluacja modelu**
- Metryki oceny (np. accuracy, F1-score, confusion matrix)

- Analiza błędów i trudnych przypadków

- Porównanie wyników dla różnych reprezentacji audio

- Ewaluacja modelu ensemble w porównaniu z pojedynczymi modelami

**7. Wnioski i przyszłe kierunki**
- Podsumowanie wyników

- Analiza efektywności różnych reprezentacji audio

- Ograniczenia projektu

- Propozycje dalszych ulepszeń i badań

- Udoskonalone proponowane narzędzia

**Język programowania:** Python

**Biblioteki do przetwarzania audio:**

***librosa*** - do ekstrakcji cech audio i tworzenia spektrogramów Mel'a

***tensorflow-io**** - do efektywnego przetwarzania danych audio w TensorFlow

***pyAudioAnalysis**** - do ekstrakcji dodatkowych cech audio

**Biblioteki uczenia maszynowego:**

***TensorFlow*** - do implementacji modeli CNN

***Keras*** - do szybkiego prototypowania i eksperymentowania z różnymi architekturami

**Biblioteki do wizualizacji:**

***matplotlib i seaborn*** - do tworzenia wykresów i wizualizacji wyników

***librosa.display*** - do wizualizacji spektrogramów i innych reprezentacji audio

**Narzędzia do zarządzania eksperymentami:**

***TensorBoard*** - do monitorowania procesu treningu i wizualizacji wyników

***MLflow*** - do śledzenia eksperymentów i wersjonowania modeli

**Środowisko pracy:**

***Jupyter Notebook*** - do interaktywnej analizy danych i prototypowania

***Google Colab z GPU*** - do treningu modeli wymagających dużej mocy obliczeniowej

**Narzędzia do wersjonowania kodu:**
***Git i GitHub*** - do kontroli wersji i współpracy



Git i GitHub - do kontroli wersji i współpracy
