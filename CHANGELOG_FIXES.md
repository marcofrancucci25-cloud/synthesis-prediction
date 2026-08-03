# Correzione sensibilità stechiometrica — v10.11.2

- eliminate le alternative L/M costruite con una sottrazione arbitraria;
- selezionati solo rapporti centrali realmente osservati per sistemi chimicamente comparabili;
- mantenuta costante la quantità totale di precursori durante la perturbazione del rapporto;
- ricalcolate coerentemente le mmol di legante e metallo;
- scartate le alternative che non superano il gate numerico di validità;
- separata esplicitamente la sensibilità descrittiva dalle raccomandazioni dell'ottimizzatore.

Il modello predittivo congelato non è stato riaddestrato o alterato.

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

## 7. [NUOVA FUNZIONALITÀ] Screening chimico di compatibilità legante/solvente
**File:** `src/solubility.py` (nuovo), `src/engine.py`, `src/optimizer.py`, `app.py`
Segnalato dall'utente: l'ottimizzatore poteva proporre acqua come solvente
anche per leganti chiaramente lipofili/insolubili in acqua, perché non
esisteva alcun controllo di solubilità — né nel modello congelato (nessuna
feature di solubilità nello schema v8.0) né nella logica dell'ottimizzatore
(il pool di solventi è scelto solo per frequenza storica col metallo).

**Implementazione (livello euristico, non richiede retraining):**
- Stima quantitativa della solubilità in acqua tramite l'equazione
  ESOL/Delaney (2004), un modello QSPR consolidato e verificabile, calcolata
  da RDKit a partire dallo SMILES del legante già risolto altrove nell'app.
- Per gli altri solventi, un controllo più grezzo basato sull'indice di
  polarità di Snyder (scala tabulata standard in chimica analitica) rispetto
  al logP calcolato del legante — dichiarato esplicitamente più debole
  dell'ESOL, e mai usato per bloccare, solo per penalizzare.
- La penalità di solubilità entra nel punteggio dell'ottimizzatore
  (`Solubility_penalty`) con un peso proprio, non azzerabile da nessun
  obiettivo (a differenza di green/speed), perché un legante insolubile nel
  solvente proposto non è mai una sintesi fisicamente valida, qualunque sia
  l'obiettivo scelto. I pesi di ciascun obiettivo sono rinormalizzati
  automaticamente a somma 1.
- Se lo SMILES del legante non è disponibile, il controllo NON viene
  finto come superato: viene emesso un avviso esplicito sia nella singola
  predizione sia nell'ottimizzatore.

**Limite noto e dichiarato esplicitamente nel codice e nell'interfaccia:**
ESOL non ha un termine di punto di fusione e sovrastima sistematicamente la
solubilità di acidi aromatici rigidi e simmetrici (es. l'acido tereftalico,
H2BDC, il legante più usato negli esempi di questa app: ESOL stima logS
-1,8 "probabilmente solubile" contro un valore sperimentale reale di circa
-4, "poco solubile"). Il controllo intercetta bene i casi netti (es. una
porfirina molto lipofila in acqua pura) ma NON sostituisce il giudizio
chimico per questa classe di leganti. Il messaggio mostrato in interfaccia
lo dichiara esplicitamente.

**Verificato:** con un legante lipofilo di test (porfirina tetracarbossilica),
l'acqua passa da 10/10 proposte dell'ottimizzatore (senza il controllo) a
0/10 (con il controllo attivo), sostituita da etanolo/metanolo. 20 nuovi
test dedicati (`test_solubility_feature.py`), tutti PASS; nessuna
regressione sulle 121 verifiche precedenti (61 test ufficiali + 47 + 13,
con un aggiornamento a `verify_fixes.py` per riflettere il nuovo avviso
legittimo quando lo SMILES manca).

## 8. [MODIFICA INTERFACCIA] Ricerca letteratura: pulsante reset e riordino dei risultati
**File:** `app.py`
Richiesto dall'utente. Due modifiche alla pagina "Literature search":
- Aggiunto un pulsante "Reset search" accanto a "Search literature" (stesso
  pattern già usato per predizione e ottimizzatore: `_reset_literature_inputs()`
  pulisce sia i risultati sia i campi del form). Per renderlo possibile, il
  form è stato convertito da `st.form(...)` a widget con `key` espliciti —
  Streamlit non permette pulsanti generici con `on_click` dentro un `st.form`.
- Ogni risultato ora mostra, in quest'ordine: **titolo** (invariato, grande e
  cliccabile) → **DOI** (`st.caption`, più piccolo del titolo) → **abstract**
  (font ridotto ma leggibile, 0.92rem, non un semplice `st.write`).
- Aggiunta protezione: titolo e abstract vengono passati da `html.escape()`
  prima di essere inseriti nel blocco `unsafe_allow_html=True` dell'abstract,
  per evitare che testo HTML/markdown imprevisto proveniente dai risultati
  di ricerca esterni (Tavily) rompa il layout della pagina.

**Verificato:** 3 nuovi test (`tests/test_literature_search_interface_v10_11_3.py`)
verificano la presenza del pulsante di reset, l'ordine esatto titolo→DOI→abstract
nel sorgente, e l'uso dell'escaping HTML. Nessuna regressione sulle 141
verifiche precedenti (61 test ufficiali + 47 + 13 + 20).
