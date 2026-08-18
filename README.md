# Faro
```
               /^\                       
              \|0|/                      
             |=====|                     
              |===|                      
              |===|                      
              |===|                /|    
        ^^^   |===|               / |    
      ^^^^^^^ |===|              /__|    
  ^^^^^^^^^^^^^^^^^^^          \_______/ 
^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~~~~~~~~~~~~
```

🇬🇧 (Voor Nederlands, zie hieronder)

This repository contains an independent open-source model for identifying potentially harmful gambling behaviour. The code to create a model like this, is called **Open Water** and is available on: https://github.com/carda-repo/openwater

The model is released under the GNU Affero General Public License v3.0 only (AGPL-3.0-only). It was trained on data obtained under the following legal framework: https://zoek.officielebekendmakingen.nl/stcrt-2021-48712.html. The project received funding from ZonMw, a Dutch public funding organisation focused on health, care, and wellbeing research (https://www.zonmw.nl/).

We aligned the model with existing regulatory data standards of the ControleDatabank of the Dutch regulater (Kansspelautoriteit). By doing so, it can be directly implemented by operators and supervisory authorities. You can find the data model here: https://kansspelautoriteit.nl/sites/default/files/ksa_cdb_datamodel_version_1-11_3_september_2024.pdf

## Get started

### What is included

- [faro.joblib](faro.joblib): the trained model artifact
- [faro.json](faro.json): metadata describing the model and feature schema
- [open_source_tool/model_utils.py](open_source_tool/model_utils.py): helpers for loading the model, creating a valid feature frame and interpreting its score
- [LICENSE](LICENSE): the GNU Affero General Public License v3.0 covering the original source code and distributed trained model artifact

### Requirements

- Python 3.11.x (tested with 3.11.9)
- Poetry

### Setup

**macOS/Linux**

```bash
poetry install
```

**Windows (PowerShell)**

```powershell
poetry env use (py -3.11 -c "import sys; print(sys.executable)")
poetry install
```

### Run an example

```bash
poetry run python -c "from open_source_tool.model_utils import load_model, build_feature_frame, predict_with_model; model = load_model('faro.joblib'); features = build_feature_frame(model); print(predict_with_model(model, features))"
```

Example output:

```python
{
    "prediction": 0,
    "score": 0.0132,
    "risk_band": "no_signal",
    "broad_screening_threshold": 0.2403,
    "f1_optimal_threshold": 0.5763
}
```

### How to interpret the output

The model was trained to identify behavioural patterns associated with temporary or indefinite self-exclusion during the subsequent target period of approximately one month. Class `1` means that such a self-exclusion was observed; class `0` means that no such self-exclusion was recorded in the training target.

The returned `score` is the output of `predict_proba(X)[:, 1]`. It is used to rank risk (a person displaying high risk should get a higher score than a person displaying low risk), but the score itself has not been calibrated and must not be interpreted as a literal probability of future self-exclusion.

The helper applies two thresholds documented in the metadata. Their numerical values were selected using validation-set predictions and subsequently evaluated on the test set. Neither threshold is universally preferred; the appropriate operational choice depends on the relative consequences of false-negative and false-positive signals:

- `score < 0.2403`: `no_signal`
- `0.2403 <= score < 0.5763`: `risk_signal`, using the broad screening threshold selected by maximizing Youden's J
- `score >= 0.5763`: `high_priority_risk_signal`, using the more selective threshold selected by maximizing F1

A signal is intended to support follow-up assessment. It is not a clinical diagnosis.

### Input requirements

The joblib contains a trained pipeline with removal of zero-variance features, standard scaling and an XGBoost classifier. It expects 180 already calculated input columns: 60 behavioural features over three lookback periods of approximately 1, 6 and 12 months. The exact schema is documented in [faro.json](faro.json).

The joblib does not calculate these features from raw ControleDatabank records. Feature engineering must be performed separately using the same definitions, units and time windows as during training.

`build_feature_frame()` creates a correctly shaped row containing zeros for all 180 features. This is only a technical smoke test. It does not represent a real player, and its output is not a meaningful risk assessment. For real use, every required feature must be populated with a correctly calculated value before calling `predict_with_model()`.

### Working with the joblib

Load the model and metadata:

```python
from open_source_tool.model_utils import load_metadata, load_model

model = load_model("faro.joblib")
metadata = load_metadata("faro.json")
```

Inspect the pipeline and the expected input columns:

```python
print(model.named_steps)
print(model.feature_names_in_)
print(len(model.feature_names_in_))  # 180
```

To make a real prediction, create a pandas DataFrame containing one player and all 180 correctly calculated features. The following example assumes that such a row has already been exported to `player_features.csv`:

```python
import pandas as pd

from open_source_tool.model_utils import predict_with_model

features = pd.read_csv("player_features.csv")
expected_columns = list(model.feature_names_in_)

if len(features) != 1:
    raise ValueError("predict_with_model() expects exactly one player row.")

missing_columns = sorted(set(expected_columns) - set(features.columns))
extra_columns = sorted(set(features.columns) - set(expected_columns))

if missing_columns or extra_columns:
    raise ValueError(
        f"Input schema mismatch. Missing: {missing_columns}; extra: {extra_columns}"
    )

features = features[expected_columns]
result = predict_with_model(model, features, metadata)
print(result)
```

`predict_with_model()` returns the model score, the corresponding risk band and a binary screening signal. It currently processes one player row at a time. The helper uses the broad-screening and F1-optimal thresholds from the metadata; calling `model.predict()` directly instead uses the classifier's default threshold of `0.5`.

You can also obtain the uncalibrated class-1 score directly:

```python
score = float(model.predict_proba(features)[0, 1])
print(score)
```

The joblib can transform a complete 180-feature row and produce a score for subsequent self-exclusion. It cannot read raw ControleDatabank tables, calculate the 180 features, explain why an individual player behaves in a particular way, or establish a clinical diagnosis or causal relationship. Please also see the DGOJ's ongoing work, which uses a different target and may offer a complementary perspective.

## License

Unless stated otherwise, the original source code and distributed trained model artifact in this repository are made available under the [GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`). Third-party dependencies remain subject to their respective licenses.

## Acknowledgements

This project would not have been possible without the support of many people and organisations:

🎓 Academic guidance — Prof. Reinout Wiers (UvA), Prof. Frenk van Harreveld (UvA), Prof. Johan Bollen (UvA), Tinka Beemsterboer (UvA) and Prof. Anneke Goudriaan (Amsterdam UMC) for their advice and early support.

💶 Funding and institutional support — ZonMw, particularly Shalini Harinarain and Emma Passchier for their trust and positive spirit.

🛠️ Data and implementation — The Dutch Gambling Authority (Kansspelautoriteit), particularly Judith Kas and her colleagues Floor van Bakkum, Flóra Felső, Niels Heijnekamp, Terra Obels and Peter Eerligh for their data-engineering expertise, enthusiasm and continued work on implementation.

🇪🇸 International collaboration — The Spanish gambling regulator DGOJ, particularly José Antonio Salmerón Garrido. Aligning with their variables makes future comparison and migration between models easier, and this connection is the reason giving this tool a Spanish name. Their forthcoming model will be revolutionary, and will provide a much-needed complementary signal to facilitate comparisons between approaches. I look forward to their publications.

🎰 Data providers — The gambling operators that complied with the regulatory data requirements that made this work possible.

❤️ Nienke - for everything.

Any feedback is welcome, I'm easy to find via the internet. 

Charles de Leau


---


<p align="right">🇳🇱</p>

# Opensourcemodel voor gokrisico's

Deze repository bevat een onafhankelijk opensourcemodel voor het identificeren van mogelijk schadelijk gokgedrag. De code om een model als hieronder te trainen is open source beschikbaar op: https://github.com/carda-repo/openwater

Het model wordt beschikbaar gesteld onder uitsluitend versie 3.0 van de GNU Affero General Public License (`AGPL-3.0-only`). Het is getraind op gegevens die zijn verkregen binnen het volgende wettelijke kader: https://zoek.officielebekendmakingen.nl/stcrt-2021-48712.html. Het project ontving financiering van ZonMw, een Nederlandse publieke financieringsorganisatie die zich richt op onderzoek naar gezondheid, zorg en welzijn (https://www.zonmw.nl/).

Doordat het model aansluit bij de bestaande standaarden voor toezichtgegevens van de ControleDatabank van de Ksa, kan het eenvoudiger door Nederlandse kansspelaanbieders en toezichthouders worden geïmplementeerd. Het datamodel is hier te vinden: https://kansspelautoriteit.nl/sites/default/files/ksa_cdb_datamodel_version_1-11_3_september_2024.pdf

## Aan de slag

### Wat is inbegrepen

- [faro.joblib](faro.joblib): het getrainde modelbestand
- [faro.json](faro.json): metadata met een beschrijving van het model en het schema van de invoerkenmerken
- [open_source_tool/model_utils.py](open_source_tool/model_utils.py): hulpfuncties voor het laden van het model, het maken van een geldige invoertabel en het interpreteren van de score
- [LICENSE](LICENSE): de GNU Affero General Public License v3.0 voor de oorspronkelijke broncode en het gedistribueerde getrainde modelartefact

### Vereisten

- Python 3.11.x (getest met 3.11.9)
- Poetry

### Installatie

**macOS/Linux**

```bash
poetry env use python3.11
poetry install
```

**Windows (PowerShell)**

```powershell
poetry env use (py -3.11 -c "import sys; print(sys.executable)")
poetry install
```

Hiermee zoekt de Windows Python Launcher het exacte pad naar Python 3.11 op en geeft dit door aan Poetry.

### Een voorbeeld uitvoeren

```bash
poetry run python -c "from open_source_tool.model_utils import load_model, build_feature_frame, predict_with_model; model = load_model('faro.joblib'); features = build_feature_frame(model); print(predict_with_model(model, features))"
```

Voorbeelduitvoer:

```python
{
    "prediction": 0,
    "score": 0.0132,
    "risk_band": "no_signal",
    "broad_screening_threshold": 0.2403,
    "f1_optimal_threshold": 0.5763
}
```

### De uitvoer interpreteren

Het model is getraind om gedragspatronen te herkennen die samenhangen met tijdelijke of permanente zelfuitsluiting in de daaropvolgende doelperiode van ongeveer één maand. Klasse `1` betekent dat zo'n zelfuitsluiting is waargenomen; klasse `0` betekent dat in het trainingsdoel geen zelfuitsluiting is geregistreerd.

De teruggegeven `score` is de uitvoer van `predict_proba(X)[:, 1]`. De score wordt gebruikt om risico te rangschikken, maar is niet gekalibreerd en mag niet worden geïnterpreteerd als een letterlijke kans op toekomstige zelfuitsluiting.

De hulpfunctie gebruikt twee grenswaarden die in de metadata zijn vastgelegd. De numerieke waarden zijn geselecteerd met voorspellingen op de validatieset en vervolgens geëvalueerd op de testset. Geen van beide grenswaarden heeft universeel de voorkeur; de geschikte operationele keuze hangt af van de relatieve gevolgen van foutnegatieve en foutpositieve signalen:

- `score < 0.2403`: `no_signal`
- `0.2403 <= score < 0.5763`: `risk_signal`, met de brede screeningsgrens die is geselecteerd door Youden's J te maximaliseren
- `score >= 0.5763`: `high_priority_risk_signal`, met de selectievere grens die is geselecteerd door F1 te maximaliseren

`prediction` is `1` voor beide risicosignalen en `0` voor `no_signal`. Een signaal is bedoeld ter ondersteuning van een vervolgbeoordeling. Het is geen klinische diagnose.

### Vereisten voor de invoer

De joblib bevat een getrainde pipeline die kenmerken zonder variantie verwijdert, de kenmerken standaardiseert en vervolgens een XGBoost-classificatiemodel toepast. De pipeline verwacht 180 vooraf berekende invoerkolommen: 60 gedragskenmerken over drie terugkijkperiodes van ongeveer 1, 6 en 12 maanden. Het exacte schema staat in [faro.json](faro.json).

De joblib berekent deze kenmerken niet uit ruwe gegevens uit de ControleDatabank. De kenmerken moeten afzonderlijk worden berekend met dezelfde definities, eenheden en periodes als tijdens de training.

`build_feature_frame()` maakt een invoerrij met de juiste 180 kolommen en vult deze allemaal met nul. Dit is uitsluitend een technische rooktest. De rij stelt geen echte speler voor en de uitvoer is geen betekenisvolle risico-inschatting. Voor werkelijk gebruik moet ieder vereist kenmerk correct worden berekend en ingevuld voordat `predict_with_model()` wordt aangeroepen.

### Werken met de joblib

Laad het model en de metadata:

```python
from open_source_tool.model_utils import load_metadata, load_model

model = load_model("faro.joblib")
metadata = load_metadata("faro.json")
```

Bekijk de pipeline en de verwachte invoerkolommen:

```python
print(model.named_steps)
print(model.feature_names_in_)
print(len(model.feature_names_in_))  # 180
```

Maak voor een echte voorspelling een pandas-DataFrame met één speler en alle 180 correct berekende kenmerken. Het volgende voorbeeld gaat ervan uit dat zo'n rij al is geëxporteerd naar `player_features.csv`:

```python
import pandas as pd

from open_source_tool.model_utils import predict_with_model

features = pd.read_csv("player_features.csv")
expected_columns = list(model.feature_names_in_)

if len(features) != 1:
    raise ValueError("predict_with_model() verwacht precies één spelersrij.")

missing_columns = sorted(set(expected_columns) - set(features.columns))
extra_columns = sorted(set(features.columns) - set(expected_columns))

if missing_columns or extra_columns:
    raise ValueError(
        f"Het invoerschema klopt niet. Ontbrekend: {missing_columns}; extra: {extra_columns}"
    )

features = features[expected_columns]
result = predict_with_model(model, features, metadata)
print(result)
```

`predict_with_model()` geeft de modelscore, de bijbehorende risicoband en een binair screeningssignaal terug. De hulpfunctie verwerkt momenteel één spelersrij per keer. De functie gebruikt de brede screeningsgrens en de F1-optimale grens uit de metadata; een rechtstreekse aanroep van `model.predict()` gebruikt daarentegen de standaardgrens van `0.5` van het classificatiemodel.

De ongekalibreerde score voor klasse `1` kan ook rechtstreeks worden opgevraagd:

```python
score = float(model.predict_proba(features)[0, 1])
print(score)
```

De joblib kan een volledige rij met 180 kenmerken verwerken en een score voor daaropvolgende zelfuitsluiting produceren. Het bestand kan geen ruwe tabellen uit de ControleDatabank inlezen, de 180 kenmerken berekenen, verklaren waarom een individuele speler bepaald gedrag vertoont, of een klinische diagnose of causaal verband vaststellen. Bekijk ook het lopende werk van de DGOJ, dat een andere doelvariabele gebruikt en mogelijk een aanvullend perspectief biedt.

## Licentie

Tenzij anders vermeld, worden de oorspronkelijke broncode en het gedistribueerde getrainde modelartefact in deze repository beschikbaar gesteld onder uitsluitend versie 3.0 van de [GNU Affero General Public License](LICENSE) (`AGPL-3.0-only`). Voor externe afhankelijkheden blijven de bijbehorende eigen licenties gelden.

## Dankwoord

Dit project zou niet mogelijk zijn geweest zonder de steun van vele mensen en organisaties:

🎓 Academische begeleiding — prof. dr. Reinout Wiers (UvA), prof. dr. Frenk van Harreveld (UvA), prof. dr. Johan Bollen (UvA), Tinka Beemsterboer (UvA) en prof. dr. Anneke Goudriaan (Amsterdam UMC), voor hun advies, vertrouwen en inspiratie.

💶 Financiering en institutionele steun — ZonMw, in het bijzonder Shalini Harinarain en Emma Passchier voor hun positieve insteek en projectbegeleiding.

🛠️ Data en implementatie — De Kansspelautoriteit (Ksa), in het bijzonder Judith Kas en haar collega's Floor van Bakkum, Flóra Felső, Niels Heijnekamp, Terra Obels en Peter Eerligh, voor hun expertise op het gebied van data-engineering, hun enthousiasme en hun voortdurende inzet bij de implementatie.

🇪🇸 Internationale samenwerking — De Spaanse kansspelautoriteit DGOJ, in het bijzonder José Antonio Salmerón Garrido. Door aan te sluiten bij hun variabelen worden toekomstige vergelijkingen en migraties tussen modellen eenvoudiger. Hun toekomstige model zal revolutionair zijn, omdat die een nieuw signaal biedt waarmee vergelijkingen tussen benaderingen voor het eerst mogelijk gemaakt worden. Ik kijk uit naar hun publicaties.

🎰 Dataleveranciers — De kansspelaanbieders die hebben voldaan aan de wettelijke vereisten voor gegevensverstrekking en dit werk daarmee mogelijk hebben gemaakt.

❤️ Nienke - voor alles.

Feedback is van harte welkom, ik ben goed te vinden op internet.

Charles de Leau

(for English 🇬🇧, scroll up)