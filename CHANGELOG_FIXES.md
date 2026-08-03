# Correzioni applicate — v10.11.1 (patch di test)

Le seguenti correzioni sono state applicate al codice sorgente v10.11.0 in risposta
al report di test (Report_Test_MOF_Synthesis_Assistant.docx) e a un secondo giro di
approfondimento mirato alla copertura delle variabili di sintesi. Tutte le modifiche
sono state verificate contro:
- la test suite originale del progetto (tests/), 57/57 PASS invariati;
- una batteria di 47 test mirati (deep_test.py), 47/47 PASS;
- 13 verifiche specifiche sulle correzioni 1-4 (verify_fixes.py), 13/13 PASS;
- verifiche mirate aggiuntive per le correzioni 5 e 6, tutte confermate corrette.

## 1. [BUG, priorità Alta] Etichette "Strategy" sovrascritte nell'ottimizzatore
**File:** `src/optimizer.py`
Le quattro etichette di strategia (Maximum probability, Best hybrid score,
Strongest successful precedent, Resource-conscious) venivano assegnate in
sequenza sullo stesso DataFrame senza controllare sovrapposizioni: se una riga
vinceva più di una categoria, solo l'ultima etichetta assegnata sopravviveva,
facendo sparire silenziosamente "Best hybrid score" dall'elenco mostrato
all'utente quando coincideva con un'altra categoria.
**Fix:** le etichette vengono ora accumulate per riga (es. "Best hybrid score
& Resource-conscious") invece di sovrascriversi.

## 2. [RISCHIO, priorità Bassa] Crash su valori numerici non validi
**File:** `src/engine.py`, `src/optimizer.py`
Un valore non numerico (es. la stringa letterale "unknown" per lo stato di
ossidazione) che raggiungesse `predict()` causava un'eccezione non gestita.
**Fix:** coercizione difensiva (`pd.to_numeric(..., errors="coerce")`) sulle
colonne numeriche dello schema del modello, prima di ogni chiamata al modello,
sia nel predittore sia nell'ottimizzatore.

## 3. [RISCHIO, priorità Media] Fallback silenzioso su solventi "ammessi" inesistenti
**File:** `src/optimizer.py`, `app.py`
Se `allowed_solvents` non trovava corrispondenze nel pool specifico per il
metallo, l'ottimizzatore usava comunque il valore richiesto alla lettera,
senza alcun avviso, anche quando quel solvente non compariva in nessun
record sperimentale.
**Fix:** il metadata dell'ottimizzatore include ora `warnings`, mostrato in
interfaccia, che distingue "solvente noto altrove ma non per questo metallo"
da "solvente mai osservato in nessun record".

## 4. [RISCHIO, priorità Media] Nessun controllo tra legante e famiglia dichiarata
**File:** `src/engine.py`, `app.py`
La famiglia del legante è un campo modificabile liberamente dall'utente e
influisce direttamente sulla predizione, ma non veniva mai confrontata con il
nome del legante stesso (es. H2BDC dichiarato "Pyridyl/N-donor" invece di
"Carboxylate" cambiava la probabilità di cristallinità dal 38% al 61%, senza
alcun avviso).
**Fix:** nuova funzione `_family_consistency()` in `applicability()`: se la
famiglia dichiarata non corrisponde a quella inferita dal nome del legante,
il punteggio di dominio di applicabilità viene penalizzato e l'etichetta non
può risultare "Inside domain". L'interfaccia mostra un avviso esplicito. Il
caso base coerente (H2BDC → Carboxylate) resta invariato.

## 5. [BUG, priorità Media — non raggiungibile dalla UI attuale] keep_solvent ignorato in presenza di allowed_solvents
**File:** `src/optimizer.py`
Impostando insieme `keep_solvent=True` e un vincolo `allowed_solvents`
incompatibile, l'ottimizzatore ignorava silenziosamente la richiesta di
mantenimento e proponeva un solvente diverso, senza alcun avviso. Il caso
analogo con `banned_solvents` invece già sollevava correttamente un errore.
**Nota:** `allowed_solvents` non è oggi esposto nel form Streamlit, quindi
questo bug non era raggiungibile dall'app pubblicata così com'è, ma resta un
difetto reale della funzione `optimize_joint()`.
**Fix:** riordinata la logica in modo che `keep_solvent`/`keep_additive`
vengano applicati per ultimi, dopo il filtro allowed/banned, sollevando un
errore esplicito e specifico in caso di reale contraddizione.

## 6. [LACUNA DI COPERTURA, priorità Media] Il metodo di sintesi non era mai richiesto all'utente
**File:** `app.py`
Indagine nata dalla domanda "il modello considera tutte le variabili?". Nei
dataset esiste una colonna `Procedura_Sintetica` (Solvothermal ~66% dei
record, Hydrothermal, Room Temperature, Microwave, Sonochemical,
Precipitation) mai richiesta nel form. Verificato empiricamente che: (a) il
classificatore a 3 classi la ignora completamente — è assente dallo schema
delle feature del modello congelato, nessuna predizione cambia; (b) il
sotto-modello KNN dei "precedenti positivi" nell'ottimizzatore la usa già
internamente, ma riceveva sempre il valore fittizio "Unknown" (mai osservato
in training) perché nessuno la passava mai, alterando silenziosamente quale
precedente sperimentale viene mostrato come supporto.
**Fix:** aggiunto un menu opzionale "Synthesis procedure" nel form, con nota
di trasparenza esplicita che il metodo influenza solo l'evidenza dei
precedenti nell'ottimizzatore, non la predizione di cristallinità. Se
l'utente non lo specifica, il comportamento resta identico a prima.

## Osservazioni riportate ma NON modificate nel codice
Vedi il messaggio di accompagnamento per il dettaglio; in sintesi:
- Sensitività non monotona a Tempo_ore e a forma a "U" per Rapporto_LM:
  pattern appresi dal modello congelato, da verificare con un chimico
  esperto rispetto alla letteratura MOF; non "correggibili" senza
  ri-addestramento.
- Chimica del legante rappresentata solo da n-grammi testuali + famiglia
  categorica, non da descrittori molecolari calcolati (dimensione, denticità,
  pKa) pur avendo RDKit disponibile — limite già dichiarato nelle note dello
  schema del modello stesso.
- Variabili mai registrate nei dati sorgente (pH, atmosfera inerte,
  pressione, velocità di agitazione, rampa di riscaldamento/raffreddamento,
  invecchiamento, purezza/particle size dei reagenti): nessun modello,
  per quanto buono, potrebbe usarle senza una nuova raccolta dati.
- Il possibile disallineamento tra i campi mmol_Legante/mmol_Sale e
  Rapporto_LM nel form (tre widget indipendenti che non si aggiornano a
  vicenda) è già intercettato dal gate di validità esistente
  (`prediction_validity`), che penalizza e segnala il caso; non è stata
  modificata la sincronizzazione reattiva dei widget Streamlit per non
  introdurre una regressione non verificabile in questo ambiente di test.

## File modificati
- `src/engine.py`
- `src/optimizer.py`
- `app.py`
- `tests/test_positive_recommendation.py` (stringa di versione attesa aggiornata a 10.6.1)

## File di test aggiunti (non necessari in produzione, utili per revisione)
- `deep_test.py` — batteria dei 47 test del report iniziale
- `verify_fixes.py` — 13 verifiche mirate sulle correzioni 1-4
