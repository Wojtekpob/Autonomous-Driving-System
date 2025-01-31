# Dokumentacja Systemu Sterowania Autonomicznym Pojazdem w Symulatorze CARLA

## Wprowadzenie

W ostatnich latach branża motoryzacyjna przechodzi dynamiczną transformację, napędzaną rozwojem systemów wspomagających kierowcę (ADAS - Advanced Driver Assistance Systems) oraz dążeniem do pełnej autonomizacji pojazdów. Kluczową rolę w tym procesie odgrywa postęp w dziedzinie sztucznej inteligencji, w szczególności wizji komputerowej, która umożliwia tworzenie coraz bardziej zaawansowanych algorytmów przetwarzania obrazu.

Jednym z istotnych elementów innowacji jest detekcja pasów ruchu, stanowiąca fundament bezpiecznego i skutecznego sterowania pojazdem. Niniejsze repozytorium zawiera implementację autonomicznego systemu sterowania pojazdem w symulatorze CARLA, w którym wykorzystano sieć neuronową **LaneNet** do detekcji pasów ruchu oraz algorytm **Model Predictive Control (MPC)** do sterowania pojazdem.

Niniejsze repozytorium stanowi podstawę pracy inżynierskiej pod tytułem "Projekt i implementacja algorytmu detekcji pasa ruchu", wykonanej na wydziale Elektroniki i Technik Informacyjnych Politechniki Warszawszkiej.

Autor: Wojciech Pobocha

## Struktura repozytorium
```
📂 project_root/
├── 📂 carla/ # Obsługa symulatora CARLA
├── 📂 model/ # Implementacja sieci LaneNet
├── 📂 tests/ # Testy jednostkowe
├── 📝 config.yml # Plik konfiguracyjny
├── 🐍 autonomous_driving_system.py # Główny moduł autonomicznego sterowania
├── 🐍 carla_client.py # Klient komunikujący się z symulatorem CARLA
├── 🐍 lane_detection.py # Implementacja detekcji pasów ruchu
├── 🐍 mpc_controller.py # Algorytm MPC do sterowania pojazdem
├── 🐍 path_planning.py # Planowanie trajektorii jazdy
├── 🐍 visualization.py # Wizualizacja wyników detekcji i trajektorii
```

## Opis modułów

### 1. **Detekcja pasów ruchu - LaneNet**

Moduł implementujący detekcję pasów ruchu na podstawie sieci neuronowej **LaneNet**. Umożliwia segmentację semantyczną obrazu w celu identyfikacji pasów ruchu. Dane wejściowe pochodzą z symulatora CARLA, a wyniki są wykorzystywane do dalszego przetwarzania w procesie sterowania.

Pliki powiązane:

- `lane_detection.py` - implementacja algorytmów detekcji pasów ruchu.
- `model/` - kod odpowiedzialny za wczytanie oraz użycie modelu LaneNet.

### 2. **Planowanie trasy**

Moduł odpowiedzialny za wyznaczanie optymalnej ścieżki dla pojazdu na podstawie wykrytych pasów ruchu oraz innych dostępnych informacji.

Pliki powiązane:

- `path_planning.py` - implementacja algorytmów planowania trasy.

### 3. **Sterowanie - Model Predictive Control (MPC)**

Implementacja sterowania autonomicznym pojazdem za pomocą **Model Predictive Control (MPC)**, co pozwala na dynamiczne dostosowanie trajektorii pojazdu w odpowiedzi na zmieniające się warunki drogowe.

Pliki powiązane:

- `mpc_controller.py` - implementacja MPC.

### 4. **Interakcja z symulatorem CARLA**

Moduł odpowiedzialny za komunikację z symulatorem **CARLA**, pobieranie obrazów oraz danych z czujników w celu analizy i dalszego przetwarzania.

Pliki powiązane:

- `carla_client.py` - obsługa komunikacji z CARLA.

### 5. **Wizualizacja wyników**

Moduł odpowiedzialny za wizualizację wykrytych pasów ruchu oraz trajektorii pojazdu.

Pliki powiązane:

- `visualization.py` - generowanie wizualizacji.

## Instalacja i Uruchomienie

### Wymagania Systemowe

- **Python 3.8.10**
- **Symulator CARLA** ([Pobierz z oficjalnej strony](https://carla.org/))
- **Biblioteki Python** (lista w `requirements.txt`)

### Instalacja

1. **Pobranie i instalacja CARLA**
   - Pobierz CARLA: [https://carla.org/](https://carla.org/)
   - Rozpakuj pliki i przejdź do katalogu CARLA

2. **Uruchomienie serwera CARLA**
   - Otwórz terminal i przejdź do katalogu CARLA:
     ```bash
     cd /ścieżka/do/CARLA
     ```
   - Uruchom symulator w trybie serwera:
     ```bash
     ./CarlaUE4.exe -carla-server -windowed -ResX=480 -ResY=320 -quality-level=Low -benchmark -fps=10
     ```

3. **Sklonowanie repozytorium i instalacja zależności**
   - Pobierz repozytorium:
     ```bash
     git clone https://github.com/Wojtekpob/Autonomous-Driving-System.git
     cd Autonomous-Driving-System
     ```
   - Zainstaluj wymagane biblioteki:
     ```bash
     pip install -r requirements.txt
     ```

### Uruchomienie Systemu

Po uruchomieniu serwera CARLA wykonaj poniższą komendę:
```bash
python autonomous_driving_system.py
```
System połączy się z CARLA, wykryje pasy ruchu i wyznaczy trajektorię pojazdu.

## Szkolenie i Walidacja Modelu LaneNet

Model LaneNet pochodzi z repozytorium: [LaneNet - An Instance Segmentation Approach](https://github.com/ShenhanQian/Lane_Detection-An_Instance_Segmentation_Approach). Implementacja LaneNet oraz folder `model/` zostały pobrane z tego źródła.

Aby wytrenować model LaneNet, postępuj zgodnie z instrukcjami zawartymi w powyższym repozytorium.

Po zakończeniu treningu, umieść wytrenowaną sieć w `config.yml` pod kluczem `lane_detection: ckpt_path:`.

## Testy i Walidacja Systemu

Repozytorium zawiera zestaw testów:
```bash
python -m unittest discover tests
```