# Used Car Price Estimator
En webbasert applikasjon som estimerer prisen på brukte biler ved hjelp av maskinlæring. 
Brukeren fyller inn bilinformasjon, og modellen returnerer et prisestimat i USD eller NOK.

- Predikerer bruktbilpriser basert på historiske data
- Maskinlæringsmodell trent i Python (scikit-learn)
- FastAPI backend med /predict og /schema endepunkt
- Web-frontend i HTML/CSS/JS
- Automatisk valuta-konvertering (USD/NOK)
- Data-validering og kategorinormalisering

Modellen bruker:
- ColumnTransformer (StandardScaler + OneHotEncoder)
- HistGradientBoostingRegressor

Hvorda kjøre:
# 1. Clone repo
git clone <repo-url>

# 2. Installer avhengigheter
pip install -r requirements.txt

# 3. Tren modellen (valgfritt)
python -m src.train

# 4. Start backend (API)
uvicorn src.api:app --reload

# 5. Start frontend
python -m http.server 5500

# 6 Deretter åpne i browser: 
http://localhost:5500/web

Videre forbedringer:
- Hente ferske bruktbilpriser live
- Bedre feature engineering (HP, motorvolum, region)
- Graph-basert prisvisualisering
- Deploy til Cloud
