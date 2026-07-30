# MOF Synthesis Assistant

Applicazione Streamlit per interrogare un database di sintesi di Metal–Organic Frameworks, stimare l'esito di nuove condizioni sperimentali e proporre condizioni alternative ordinate per probabilità di cristallinità.

## Moduli

- **Prediction Engine**: classificazione calibrata in fallimento, prodotto amorfo/scarsamente cristallino o MOF cristallino.
- **Knowledge Engine**: ricerca e analisi delle sintesi presenti nel database.
- **Applicability Domain**: stima della vicinanza della nuova sintesi ai dati di sviluppo.
- **Local Explainability**: analisi di sensibilità feature-by-feature.
- **Optimizer Engine**: esplorazione locale di solvente, additivo, temperatura, tempo e rapporto L/M.
- **Validation Dashboard**: risultati dell'External Test finale.

## Prestazioni esterne

L'External Test comprende 269 sintesi con ligandi completamente separati dal Development Set.

| Metrica | Valore |
|---|---:|
| Accuracy | 0.810 |
| Balanced Accuracy | 0.820 |
| Macro-F1 | 0.801 |
| MCC | 0.704 |
| Log-loss | 0.507 |
| Multiclass Brier | 0.288 |
| ECE | 0.060 |

## Deploy su Streamlit Community Cloud

1. Caricare **tutto il contenuto di questa cartella** nella root del repository GitHub.
2. Accedere a Streamlit Community Cloud e scegliere **Create app**.
3. Selezionare il repository `marcofrancucci25-cloud/mof-synthesis-app`.
4. Impostare il branch `main` e il main file path `app.py`.
5. Scegliere Python 3.13, coerente con la versione usata per serializzare il modello.
6. Avviare il deploy.

Il file `requirements.txt` è nella root, come richiesto da Streamlit Community Cloud.

## Esecuzione locale

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Controllo rapido

```bash
python tests/smoke_test.py
```

## Struttura

```text
app.py
requirements.txt
.streamlit/config.toml
src/
  assets.py
  constants.py
  engine.py
  ui.py
data/
  knowledge_database.csv
models/
  MOF_RandomForest_Calibrated_v6_3.joblib
  applicability_domain_v7_0.joblib
  feature_schema.json
reports/
  external_test_metrics.json
  external_class_metrics.csv
  external_confusion_matrix.csv
  external_predictions.csv
```

## Uso scientifico e limiti

Il sistema è uno strumento di supporto decisionale. Le probabilità non garantiscono l'esito sperimentale. Le proposte devono essere controllate per compatibilità chimica, solubilità, stabilità, sicurezza e fattibilità. Le condizioni fuori dominio devono essere considerate esplorative.

## Citazione

Una citazione formale verrà aggiunta dopo la pubblicazione del manoscritto associato.

## Licenza

Codice distribuito con licenza MIT. Il database e il modello sono inclusi per l'utilizzo dell'applicazione; verificare i diritti sulle fonti bibliografiche prima di redistribuire dati derivati.
