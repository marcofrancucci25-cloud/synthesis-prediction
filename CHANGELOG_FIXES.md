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

## 9. [NUOVA FUNZIONALITÀ] Coerenza fisica temperatura/solvente/tipo di vaso
**File:** `src/vessel_conditions.py` (nuovo), `src/engine.py`, `src/optimizer.py`, `app.py`
Primo elemento della lista di "prossimi parametri per l'ottimizzatore"
discussa con l'utente. Fino a ora, l'ottimizzatore poteva proporre una
temperatura superiore al punto di ebollizione del solvente scelto (es. 150°C
in acqua) senza mai segnalare che questo richiede un vaso sigillato
(autoclave/vial solvotermico), non un pallone a riflusso aperto.

**Implementazione:**
- Punti di ebollizione normali (1 atm) tabulati per i solventi già usati
  nell'app; per le miscele (34% dei record storici, es. "DMF/H2O") si usa
  conservativamente il componente col punto di ebollizione più basso.
- Nuova colonna informativa `Requires_Sealed_Vessel` nei risultati
  dell'ottimizzatore. **Deliberatamente NON penalizza** `Optimization_score`:
  una sintesi solvotermica non è un difetto, è anzi il metodo più comune e
  spesso preferibile per i MOF — è solo un'informazione che serve per
  scegliere la vetreria giusta.
- Nella singola predizione, riutilizzato lo stesso pattern del controllo
  famiglia/legante già esistente: se l'utente dichiara esplicitamente
  `Procedura_Sintetica = "Room Temperature"` o `"Precipitation"` ma la
  temperatura inserita supera il punto di ebollizione del solvente scelto,
  viene mostrato un avviso di incoerenza tra i due campi dichiarati (non
  tocca il punteggio di dominio di applicabilità, che misura
  l'estrapolazione del modello, non la coerenza fisica dell'input).
- Solvente sconosciuto o temperatura mancante → `requires_sealed_vessel=None`
  (non `False`): l'assenza di dati non diventa mai un falso "va tutto bene".

**Verificato:** 17 nuovi test dedicati (`test_vessel_conditions_feature.py`),
tutti PASS. Nessuna regressione sulle 158 verifiche precedenti (64 test
ufficiali + 47 + 13 + 20).

## 10. [NUOVA FUNZIONALITÀ] Compatibilità pKa modulatore/legante ("modulazione competitiva")
**File:** `src/modulator_chemistry.py` (nuovo), `src/engine.py`, `src/optimizer.py`, `app.py`
Secondo elemento della lista discussa con l'utente. Nella sintesi modulata di
MOF, un acido monotopico (es. acido acetico, benzoico) compete col legante
per i siti di coordinazione, rallentando la nucleazione — ma perché funzioni
serve un'acidità comparabile tra modulatore e legante. Fino a ora l'additivo
era scelto dall'ottimizzatore solo per frequenza storica, senza alcun
controllo chimico.

**Implementazione (deliberatamente più cauta di solubilità e vaso di reazione):**
- pKa reali e tabulati per i modulatori/additivi effettivamente presenti nei
  dati (acido acetico 4.76, benzoico 4.20, formico 3.75, TFA 0.3, HCl/HNO3
  come acidi forti). Distingue esplicitamente i ruoli: modulatore acido,
  base (es. trietilammina — deprotona, non compete), co-legante (es. BDC/BTC
  usati come secondo linker, non come modulatore), additivo inerte.
- pKa rappresentativo per famiglia di legante (non per il legante specifico:
  nessun predittore di pKa da SMILES è disponibile in questo ambiente),
  con valori di letteratura per membri tipici di ciascuna famiglia.
- **Bug trovato e corretto durante lo sviluppo, prima della consegna**: il
  campo Famiglia_Legante arriva all'ottimizzatore già canonicalizzato al
  vocabolario interno di training (es. "Carbossilati aromatici"), non
  all'etichetta pubblica ("Carboxylate") — la tabella dei pKa inizialmente
  non lo gestiva, producendo silenziosamente solo il "ruolo" del modulatore
  senza mai calcolare il verdetto vero. Risolto riutilizzando
  `chem.canonicalize_family()` (la stessa funzione già usata da `build_row`)
  invece di duplicare la mappatura, così le due rappresentazioni restano
  sempre sincronizzate. Aggiunto un test di regressione dedicato.
- **Puramente informativo, mai una penalità nel punteggio**: a differenza
  della solubilità, qui la scienza è più qualitativa (guida di letteratura,
  non una legge fisica) e il pKa del legante è solo un'approssimazione per
  famiglia — sarebbe scorretto trattarla come un fattore di ranking con lo
  stesso peso di un dato fisico reale.

**Verificato:** 16 test dedicati (`test_modulator_chemistry_feature.py`),
tutti PASS, incluso un test di non-regressione sul bug di canonicalizzazione.
Nessuna regressione sulle 161 verifiche precedenti (64 test ufficiali + 47 +
13 + 20 + 17).

## 11. [NUOVA FUNZIONALITÀ] Screening di miscibilità acqua/solvente nelle miscele
**File:** `src/solvent_miscibility.py` (nuovo), `src/engine.py`, `src/optimizer.py`, `app.py`
Quarto e ultimo elemento della lista discussa con l'utente. Prima di
implementarlo, verificato che il pool storico di solventi campionato
dall'ottimizzatore (`_pool()`) attinge **solo** da stringhe già osservate nei
dati, mai combinazioni sintetizzate combinatorialmente — controllate tutte
le 23 miscele uniche presenti nei dati storici (DMF/H2O, H2O/EtOH,
MeOH/Toluene, ecc.): sono tutte chimicamente sensate. Il rischio reale è il
campo **Solvente in testo libero** della predizione singola, dove nulla
impedisce di scrivere una miscela come "Water/Toluene" che si separerebbe
in due fasi.

**Implementazione (portata deliberatamente stretta):**
- Controlla solo il caso acqua + solvente classicamente non miscibile
  (toluene, esano/eptano, diclorometano, cloroformio — le coppie bifasiche
  con acqua da manuale di chimica generale), più acetato di etile come
  "parzialmente miscibile" (caso più sfumato, messaggio più tenue).
  **Non** è una matrice di miscibilità completa: una coppia non segnalata
  significa "non è uno dei problemi noti controllati qui", non "confermata
  miscibile" — dichiarato esplicitamente nel codice.
- Colonna informativa `Miscibility_Flag` nei risultati dell'ottimizzatore.
  A differenza degli altri controlli informativi (vaso, modulatore), qui
  l'immiscibilità è un fatto fisico difficilmente discutibile (non un
  giudizio qualitativo): se una proposta genuinamente immiscibile sopravvive
  fino ai risultati finali, viene aggiunto anche un avviso esplicito in
  `metadata["warnings"]`, non lasciato solo come colonna da notare.
- Verificato che nel funzionamento normale (pool storico) questo avviso non
  scatta mai; scatta correttamente quando forzato tramite `keep_solvent` su
  un caso di test con "Water/Toluene".

**Verificato:** 17 nuovi test dedicati (`test_solvent_miscibility_feature.py`),
tutti PASS, incluso un test end-to-end che conferma la comparsa
dell'avviso esplicito. Nessuna regressione sulle 177 verifiche precedenti
(64 test ufficiali + 47 + 13 + 20 + 17 + 16).

---

## Riepilogo dei quattro parametri aggiunti all'ottimizzatore (voci 7-11)
| # | Parametro | Tipo di controllo | Penalizza il punteggio? |
|---|---|---|---|
| 1 | Solubilità legante/solvente (ESOL) | Quantitativo, letteratura consolidata | Sì (fisica reale) |
| 2 | Temperatura vs punto di ebollizione | Quantitativo, dati tabulati | No (informativo) |
| 3 | pKa modulatore/legante | Qualitativo, guida di letteratura | No (informativo) |
| 4 | Miscibilità acqua/solvente | Qualitativo mirato, casi classici | No, ma avviso esplicito se sopravvive ai risultati finali |

## 12. [MIGLIORAMENTO STRUTTURALE] Range numerici specifici per metallo, non più globali
**File:** `src/optimizer.py`, `app.py`
Richiesto dall'utente: "l'ottimizzatore impara meglio a valutare i parametri
di sintesi? nei MOF non sono tutti uguali". Verifica preliminare: i range
di temperatura/tempo/rapporto/volume/idratazione/stato di ossidazione che
l'ottimizzatore campiona per generare candidati venivano calcolati
sull'**intero dataset storico**, senza mai filtrare per il metallo
specifico in ottimizzazione — a differenza del pool di solventi/additivi,
che già era filtrato per metallo. Dati reali per confermare la gravità:
- **Stato di ossidazione**: Zn è sempre +2, Al sempre +3, Fe sempre +3 nei
  precursori usati; il range globale (2,0–4,0) permetteva quindi
  all'ottimizzatore di proporre stati di ossidazione chimicamente
  implausibili per un metallo specifico (es. Al a stato +2).
- **Numero di idratazione**: mediana 9 per Al, 1 per Cu, 0 per Zr — molto
  diversi tra loro; il range globale (mediana 4) non rappresentava bene
  nessuno dei tre.
- **Volume solvente**: per Zr era sempre 100 (protocollo fisso), per Al/Fe
  il dato non è nemmeno popolato — casi che richiedono una cascata di
  fallback sicura, non solo un filtro naïve.

**Implementazione:** `_quantile_bounds()` (già esistente) ricade
automaticamente sul suo argomento di fallback quando una colonna ha meno di
10 valori validi. Sfruttando questo comportamento già presente, i range ora
vengono calcolati con una cascata a tre livelli — dati specifici del
metallo → intero dataset (il comportamento precedente) → valore fisso di
sicurezza — senza dover modificare la funzione stessa. Stesso principio già
usato per il pool di solventi/additivi (`metal_success`/`pool_db`), esteso
qui ai range numerici per coerenza architetturale.
**Trasparenza:** nuovo campo `metal_specific_evidence_rows` nei metadata,
mostrato in interfaccia con un avviso quando i dati specifici del metallo
sono insufficienti (<10 righe) e i range ricadono sul dataset globale.

**Verificato con un test end-to-end concreto:** per Al, lo stato di
ossidazione proposto ora è sempre 3 (mai più 2), e la mediana del numero di
idratazione (9) riflette correttamente la chimica dell'alluminio invece del
valore globale. 9 nuovi test dedicati
(`test_metal_specific_bounds_feature.py`), tutti PASS. Nessuna regressione
sulle 194 verifiche precedenti.

**Portata dichiarata:** questa modifica condiziona i range solo per
**metallo**, non per la combinazione metallo+famiglia di legante — un
ulteriore livello di specificità (es. Zn+Bipyrazole vs Zn+Carbossilati)
resta un possibile affinamento futuro, non incluso qui per non introdurre
troppa granularità in un solo passaggio senza prima verificarne il valore.

## 13. [MIGLIORAMENTO STRUTTURALE] Range numerici specifici per metallo+famiglia di legante
**File:** `src/optimizer.py`, `app.py`
Raffinamento annunciato nella voce 12: la cascata a 3 livelli (metallo →
dataset → fisso) è ora a **4 livelli** aggiungendo il filtro per
combinazione metallo+famiglia di legante, il livello di specificità più
fine che i dati permettono.

**Verifica preliminare che ne valesse la pena:** confrontati i range
storici per lo stesso metallo ma famiglie diverse — Zn+Bipyrazole mediana
160°C contro Zn+"Non specificata" mediana 100°C; Cu+Bipyrazole 150°C contro
Cu+"Non specificata" 85°C. Differenze di 60-65°C per lo stesso metallo,
a seconda della chimica del legante: motivo sufficiente per procedere.

**Implementazione:** nuova funzione helper `_cascading_bounds()` che
incatena `_quantile_bounds()` su una sequenza di sorgenti dalla meno alla
più specifica (dataset intero → metallo → metallo+famiglia), riducendo la
ripetizione di codice rispetto alla v12 e rendendo la cascata facile da
estendere in futuro. Nessuna modifica a `_quantile_bounds()` stessa.
**Trasparenza:** nuovo campo metadata `metal_family_specific_evidence_rows`;
l'interfaccia mostra ora quale livello di specificità ha effettivamente
informato i range (metallo+famiglia, se ≥10 record; altrimenti l'avviso
già esistente sul fallback al dataset intero).

**Verificato:** per Zn, le due famiglie (Bipyrazole n=24 record, Carboxylate
n=21 record) usano ora dati distinti e producono proposte di temperatura
diverse — confermato nei metadata restituiti. Combinazioni mai osservate
(es. Ce+Phosphonate) ricadono correttamente sul livello meno specifico,
senza crash. 8 nuovi test dedicati (`test_metal_family_bounds_feature.py`),
tutti PASS. Nessuna regressione sulle 203 verifiche precedenti.

## 14. [BUG, priorità Alta] Il resolver del legante poteva proporre strutture completamente estranee
**File:** `src/literature.py`
Segnalato dall'utente con due screenshot dell'app live: cercando "3,3'-amino-
4,4'-bipyrazole", il resolver ha proposto come candidato **"Toxin C2"**
(C10H17N7O11S2, un derivato dell'acido solfamico) — una struttura senza
alcuna relazione con un bipirazolo.

**Causa radice:** quando OPSIN, PubChem e Cactus non trovano nulla, l'app
usa Tavily (motore di ricerca generico, non specializzato in chimica) come
ultima risorsa per "scoprire identificatori alternativi": cerca la query sul
web e cerca di estrarre numeri CAS, abbreviazioni tra parentesi e stringhe
tra virgolette da fino a 8 risultati. **Il problema**: estraeva questi
pattern da **qualunque punto** del testo di ciascun risultato, senza mai
verificare che quel testo parlasse davvero del legante cercato. Una pagina
web irrilevante che per puro caso conteneva sia la parola "amino" (in un
contesto completamente diverso) sia un nome tra parentesi che PubChem sa
risolvere in una molecola vera, veniva presentata come "candidato" con tanto
di nome IUPAC, formula e struttura disegnata — dando un'apparenza di
affidabilità del tutto ingannevole.

**Fix:** nuovo controllo di pertinenza (`_result_is_relevant`): un risultato
di ricerca deve contenere almeno il 60% delle parole distintive della query
(escludendo termini generici come "ligand"/"MOF"/"linker" iniettati nella
query di ricerca stessa, e numeri/locanti nudi come "3"/"4" perché troppo
comuni per essere un segnale utile) prima che qualunque identificatore
estratto da quel testo venga considerato attendibile. Usa il confronto per
sottostringa (non l'uguaglianza esatta tra parole), per riconoscere
correttamente varianti come "diamino"/"monoamino" che contengono "amino".

**Nota separata, non risolta qui:** i due nomi provati dall'utente sembrano
riferirsi alla variante **di-amino** del bipirazolo (due gruppi NH2), mentre
la libreria curata locale (`LOCAL_STRUCTURES`) contiene solo la variante
**mono-amino** (un gruppo NH2, "3-amino-4,4'-bipyrazole", C6H7N5). Ho
cercato una fonte affidabile per verificare la struttura della variante
di-amino prima di aggiungerla, ma non ho trovato un riscontro autorevole
(nessun CAS/PubChem chiaro) — non l'ho quindi inserita a mano: una struttura
"curata" ma non verificata sarebbe rischiosa quanto il bug stesso. Se l'utente
può confermare la struttura esatta (da un articolo specifico), può essere
aggiunta correttamente seguendo lo stesso schema della voce già presente.

**Verificato:** 8 nuovi test dedicati (`test_ligand_resolver_relevance_feature.py`),
inclusi 2 test end-to-end con risposta Tavily simulata (mock) che riproducono
esattamente lo scenario "Toxin C2" segnalato — ora respinto — e confermano
che un risultato genuinamente pertinente continua a funzionare. Nessuna
regressione sulle 211 verifiche precedenti.
