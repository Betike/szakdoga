# Premier League Predictor - Telepítési és futtatási útmutató

Ez az útmutató részletesen bemutatja, hogyan telepíthető és használható a Premier League Predictor alkalmazás teljesen új rendszeren. Az útmutató Windows, macOS és Linux rendszerekre is alkalmazható.

## Tartalomjegyzék

1. [Szükséges előfeltételek](#1-szükséges-előfeltételek)
2. [Python telepítése](#2-python-telepítése)
3. [Git telepítése](#3-git-telepítése)
4. [A projekt letöltése](#4-a-projekt-letöltése)
5. [Virtuális környezet létrehozása](#5-virtuális-környezet-létrehozása)
6. [Python függőségek telepítése](#6-python-függőségek-telepítése)
7. [Node.js telepítése](#7-nodejs-telepítése)
8. [Frontend telepítése](#8-frontend-telepítése)
9. [Az alkalmazás futtatása](#9-az-alkalmazás-futtatása)
10. [Hibaelhárítás](#10-hibaelhárítás)

## 1. Szükséges előfeltételek

Az alkalmazás futtatásához szükséges minimális rendszerkövetelmények:

- **Operációs rendszer**: Windows 10/11, macOS 10.15+, vagy Ubuntu 20.04+ (vagy más modern Linux disztribúció)
- **RAM**: Minimum 4 GB (8 GB ajánlott)
- **Tárhely**: Minimum 1 GB szabad hely
- **Internet kapcsolat**: A függőségek letöltéséhez és az API-kommunikációhoz

## 2. Python telepítése

### Windows

1. Látogasson el a [Python hivatalos oldalára](https://www.python.org/downloads/).
2. Töltse le és futtassa a legfrissebb Python 3.8 vagy újabb telepítőt.
3. A telepítés során jelölje be az "Add Python to PATH" opciót.
4. Válassza a "Customize installation" opciót, majd győződjön meg róla, hogy a "pip" és a "tcl/tk és IDLE" opciók ki vannak választva.
5. Kattintson a "Install Now" gombra.
6. A telepítés ellenőrzéséhez nyissa meg a parancssort (cmd) és írja be:
   ```
   python --version
   pip --version
   ```

### macOS

1. Nyissa meg a Terminalt.
2. Ha még nincs telepítve, telepítse a Homebrew csomagkezelőt:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
   ```
3. Telepítse a Pythont:
   ```
   brew install python
   ```
4. Ellenőrizze a telepítést:
   ```
   python3 --version
   pip3 --version
   ```

### Linux (Ubuntu/Debian)

1. Nyissa meg a terminált.
2. Frissítse a csomaglistát:
   ```
   sudo apt update
   ```
3. Telepítse a Pythont és a szükséges eszközöket:
   ```
   sudo apt install python3 python3-pip python3-venv python3-dev build-essential
   ```
4. Ellenőrizze a telepítést:
   ```
   python3 --version
   pip3 --version
   ```

## 3. Git telepítése

### Windows

1. Töltse le és telepítse a Git for Windows-t a [git-scm.com](https://git-scm.com/download/win) oldalról.
2. Kövesse a telepítési varázsló lépéseit, hagyja meg az alapértelmezett beállításokat.
3. A telepítés után ellenőrizze a Git elérhetőségét:
   ```
   git --version
   ```

### macOS

1. Nyissa meg a Terminalt.
2. Telepítse a Git-et Homebrew segítségével:
   ```
   brew install git
   ```
3. Ellenőrizze a telepítést:
   ```
   git --version
   ```

### Linux (Ubuntu/Debian)

1. Nyissa meg a terminált.
2. Telepítse a Git-et:
   ```
   sudo apt install git
   ```
3. Ellenőrizze a telepítést:
   ```
   git --version
   ```

## 4. A projekt letöltése

1. Nyisson egy parancssor ablakot vagy terminált.
2. Navigáljon abba a mappába, ahová a projektet szeretné letölteni:
   ```
   # Windows
   cd C:\Users\[felhasználónév]\Documents

   # macOS / Linux
   cd ~/Documents
   ```
3. Klónozza le a projektet a GitHub-ról:
   ```
   git clone [repository_url]
   ```
4. Lépjen be a projekt könyvtárába:
   ```
   cd football-predictor
```

## 5. Virtuális környezet létrehozása

A virtuális környezet segít elkülöníteni a projekt függőségeit a rendszer többi részétől.

### Windows

```
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```
python3 -m venv .venv
source .venv/bin/activate
```

A virtuális környezet aktiválása után a parancssor elején megjelenik a `(.venv)` előtag.

## 6. Python függőségek telepítése

Miután aktiválta a virtuális környezetet, telepítse a szükséges Python csomagokat:

```
pip install --upgrade pip
pip install -r requirements.txt
```

Ez telepíti az összes szükséges Python könyvtárat, beleértve az alábbiakat:
- numpy, pandas (adatelemzéshez)
- matplotlib, seaborn (vizualizációhoz)
- scikit-learn, xgboost (gépi tanuláshoz)
- torch (neurális hálózatokhoz)
- flask (API kiszolgálóhoz)
- és további segédkönyvtárakat

### Torch telepítése CUDA támogatással (opcionális)

Ha NVIDIA GPU-val rendelkezik és szeretné gyorsítani a neurális hálózat betanítását és predikciót, telepítse a PyTorch CUDA verzióját. Látogasson el a [PyTorch hivatalos oldalára](https://pytorch.org/get-started/locally/) a megfelelő telepítési parancsért az Ön CUDA verziójához.

## 7. Node.js telepítése

A frontend alkalmazás futtatásához Node.js szükséges.

### Windows

1. Látogasson el a [Node.js hivatalos oldalára](https://nodejs.org/).
2. Töltse le és futtassa az LTS (Long Term Support) telepítőt.
3. Kövesse a telepítési varázsló lépéseit az alapértelmezett beállításokkal.
4. A telepítés után ellenőrizze a Node.js és npm elérhetőségét:
   ```
   node --version
   npm --version
   ```

### macOS

1. Nyissa meg a Terminalt.
2. Telepítse a Node.js-t Homebrew segítségével:
   ```
   brew install node
   ```
3. Ellenőrizze a telepítést:
   ```
   node --version
   npm --version
   ```

### Linux (Ubuntu/Debian)

1. Nyissa meg a terminált.
2. Telepítse a Node.js-t:
   ```
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt install -y nodejs
   ```
3. Ellenőrizze a telepítést:
   ```
   node --version
   npm --version
   ```

## 8. Frontend telepítése

1. Navigáljon a frontend mappába:
   ```
   cd football-predictor-ui
   ```
2. Telepítse a szükséges Node.js csomagokat:
   ```
   npm install
   ```
3. Ellenőrizze, hogy minden függőség sikeresen települt-e.

## 9. Az alkalmazás futtatása

### Backend API indítása

1. Lépjen vissza a projekt főkönyvtárába (ha a frontend mappában tartózkodik):
   ```
   cd ..
   ```
2. Ellenőrizze, hogy a virtuális környezet aktív-e (a parancssor elején megjelenik a `(.venv)` előtag). Ha nem, aktiválja:
   ```
   # Windows
   .venv\Scripts\activate
   
   # macOS / Linux
   source .venv/bin/activate
   ```
3. Indítsa el a Flask API-t:
   ```
   python predict/run_prediction.py --model ensemble --home "Manchester United" --away "Chelsea" --json
   ```

   Megjegyzés: A fenti parancs egy egyszerű predikciót készít. A modelleket az alábbi módokon használhatja:
   - `--model xgboost`: XGBoost modell használata
   - `--model random_forest`: Random Forest modell használata
   - `--model pytorch`: Neurális hálózat modell használata
   - `--model ensemble`: Az összes modell kombinációjának használata (ajánlott)

### Frontend alkalmazás indítása

1. Nyisson egy új parancssor ablakot.
2. Navigáljon a frontend mappába:
   ```
   cd [projekt_útvonal]/football-predictor-ui
   ```
3. Indítsa el a Next.js fejlesztői szervert:
   ```
   npm run dev
   ```
4. Nyissa meg a böngészőt és navigáljon a [http://localhost:3000](http://localhost:3000) címre.

## 10. Hibaelhárítás

### Általános problémák

1. **"Python command not found" hiba**
   - Windows: Ellenőrizze, hogy a Python hozzá van-e adva a PATH környezeti változóhoz.
   - macOS/Linux: Használja a `python3` parancsot `python` helyett.

2. **Függőségek telepítési hibája**
   - Ellenőrizze az internetkapcsolatot.
   - Windows esetén: Futtassa a parancssorban adminisztrátorként.
   - Linux esetén: Telepítse a fejlesztői eszközöket: `sudo apt install python3-dev build-essential`.

3. **"No module named X" hiba**
   - Ellenőrizze, hogy aktiválta-e a virtuális környezetet.
   - Telepítse újra a függőségeket: `pip install -r requirements.txt`.

4. **Frontend indítási hibák**
   - Ellenőrizze, hogy a Node.js és npm megfelelően telepítve vannak-e.
   - Futtassa újra az `npm install` parancsot.
   - Ha a Next.js fejlesztői szerverrel kapcsolatos hibákat tapasztal, próbálja meg az `npm run dev` helyett a `npx next dev` parancsot.

### Teljesítmény problémák

- Ha a modell predikció túl lassú CPU-módban, érdemes megfontolni a CUDA-val kompatibilis GPU használatát és a PyTorch CUDA verziójának telepítését.
- Ha memóriaproblémákat tapasztal, próbáljon meg több RAM-ot biztosítani a rendszernek, vagy zárja be a nem használt alkalmazásokat.

## További információk

- A modellek a `models/` mappában találhatók.
- Az adatok a `data/` mappában vannak tárolva.
- A vizualizációk a `visualisations/` mappában találhatók.
- A predikciós logika a `predict/` mappában található.

A Premier League Predictor alkalmazás segítségével előrejelzéseket készíthet az angol labdarúgó-bajnokság mérkőzéseinek eredményeiről, különböző gépi tanulási modellek segítségével. 