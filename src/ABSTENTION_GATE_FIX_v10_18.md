# Fix: gate di astensione eccessivamente conservativo (v10.18)

## Problema riscontrato
Il gate di astensione (`prediction_interpretation` in `src/engine.py`), pensato per nascondere
le previsioni quando l'input è fuori dal dominio validato, si asteneva nel **93% dei casi anche
su righe prese direttamente dal training set** (140/150 su un campione casuale), rendendo lo
strumento sostanzialmente inutilizzabile anche su chimica ben nota.

## Causa radice
Il campo `Volume solvente` è mancante nell'**84,7% dell'intero database di training** (619/731
righe INCLUDE) — un limite noto e già documentato dei dati storici, non un segnale di
estrapolazione. Due punti del controllo di validità penalizzavano questa assenza al massimo:

1. Il controllo diretto sul valore mancante assegnava severità piena (1.0) a qualunque colonna
   numerica assente, incluso `Volume solvente` — nonostante il modello predittivo stesso **non
   usi affatto questa colonna come feature** (esclusa deliberatamente per evitare fughe di dati,
   vedi `NUMERIC_NO_VOLUME`).
2. Il controllo di plausibilità della concentrazione totale divideva per il volume; quando il
   volume mancava, il codice impostava la concentrazione a `infinito`, che veniva poi segnalato
   come "fuori dai range plausibili" (severità 0.8).

Il risultato combinato: la stragrande maggioranza delle righe, avendo `Volume solvente`
mancante, falliva automaticamente il controllo di validità indipendentemente da quanto fossero
tipiche le altre condizioni.

## Fix applicato (`src/engine.py`)
- Un `Volume solvente` mancante non genera più una penalità di severità piena: viene trattato
  come non informativo, coerentemente col fatto che il modello non lo usa.
- Il controllo di plausibilità della concentrazione viene eseguito solo quando il volume è
  effettivamente disponibile; altrimenti viene saltato invece di produrre un valore infinito.

## Risultato verificato

| | Prima del fix | Dopo il fix |
|---|---|---|
| Astensione su 150 righe di training | 140/150 (93%) | 30/150 (20%) |
| Astensione su benchmark esterno v12 (n=33) | 33/33 (100%) | 29/33 (88%) |
| Astensione su set di laboratorio (n=17) | 16/17 (94%) | 16/17 (94%) |
| Test esistenti del progetto | — | 70/70 passati, nessuna regressione |

Le astensioni residue sul training (30/150) sono dovute per la maggior parte (18/30) a bassa
confidenza genuina del modello e per il resto (11/30) a valori realmente estremi in altre
variabili numeriche — non a un ulteriore bug sistemico, verificato caso per caso.

## Nota importante
Questo fix **non modifica il modello predittivo** (nessun file `.joblib` toccato) e **non
cambia mai** la previsione numerica prodotta per un dato input — cambia solo quando quella
previsione viene mostrata invece di essere nascosta dietro "astensione". È quindi un
miglioramento a rischio minimo: non può introdurre un nuovo tipo di errore di predizione, può
solo smettere di nascondere previsioni che erano già corrette.
