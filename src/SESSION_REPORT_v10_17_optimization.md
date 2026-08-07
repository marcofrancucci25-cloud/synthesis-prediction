# Rapporto di sessione: tentativo di ottimizzazione del modello predittivo (v10.13 → v10.17)

**Stato finale: nessuna nuova versione promossa. Il modello di produzione resta `MOF_Audited_Deleaked_v10_12.joblib`.**

## 1. Obiettivo di partenza

Migliorare la capacità del modello di riconoscere correttamente sintesi Fallite (classe 0) e Amorfe (classe 1), che nella versione di produzione v10.12 avevano recall 0% su ogni set di dati genuinamente esterno testato.

## 2. Percorso seguito, in ordine cronologico

### 2.1 Espansione dati (v10.14, v10.15 — non promosse)
- Integrato **AIRES** (Rong/Yaghi, *Nature Synthesis* 2026, DOI 10.1038/s44160-025-00939-9): 4.390 record Zn+imidazolato, screening robotico, dati grezzi CSV aperti su Zenodo.
- Integrato **POMOF** (He/Cronin, *JACS* 2024, DOI 10.1021/jacs.4c09553): 303 record Zn+POM-ammina+aldeidi piridiniche, estratti a mano dalle tabelle S1-S54 della Supporting Information.
- **Risultato**: nessun miglioramento misurabile sui due benchmark esterni indipendenti (recall Amorphous fermo a 0% in entrambi i casi). Conclusione: il volume/diversità di dati da 1-2 fonti singole, per quanto ampio, non basta a superare il problema di generalizzazione.

### 2.2 Ottimizzazione architettura (v10.16, v10.17 — non promosse)
Tre interventi testati sistematicamente, ciascuno verificato sia in cross-validation raggruppata (5-fold per legante canonico) sia sui due benchmark esterni indipendenti (33 + 17 record, mai usati in training):

| Intervento | Effetto |
|---|---|
| Random Forest → HistGradientBoostingClassifier, regolarizzato (profondità 4, learning_rate 0.10) | Miglioramento consistente in CV |
| Rimozione della calibrazione (`CalibratedClassifierCV`) | Recall Amorphous 0%→100% sui set esterni, ma **crollo del recall Crystalline** (100%→0-24%) e dell'accuracy complessiva (85%→18-30%) |
| Ribilanciamento parziale delle classi (`alpha=0.75` invece di `class_weight='balanced'` pieno) | Migliora l'equilibrio in CV, ma il crollo su Crystalline nei set esterni **persiste identico** — non è quindi un problema di pesi |

**Conclusione onesta**: ogni configurazione scambia un difetto per un altro (o ignora sempre Fallito/Amorfo, o ignora sempre Cristallino) — nessuna è complessivamente migliore della produzione su tutte le metriche insieme.

## 3. Cause radice identificate (diagnosi confermata, non solo ipotesi)

Analizzando caso per caso i record esterni mal classificati:

1. **Ambiguità chimica reale** — `3-amino-4,4'-bipyrazole` e `3,5-diamino-4,4'-bipyrazole` + Zn(OAc)₂: 85 e 55 righe di training rispettivamente, con **tutte e tre le classi rappresentate alle stesse condizioni nominali** (rapporto 1:1). Non è un difetto del modello: sono condizioni genuinamente non deterministiche nei dati disponibili.

2. **Estrapolazione oltre il range di training** — `2-Methylimidazole`: il training originale (v10.5) copre rapporti legante:metallo solo fino a 4:1, ma un caso esterno reale ha rapporto 8:1 (comune nelle sintesi di tipo ZIF, dove l'eccesso di legante è normale). I modelli ad albero (RF, HGB) non estrapolano bene oltre l'intervallo osservato. **Il tranche AIRES copre già rapporti fino a 15:1 per questo legante**, ma solo a T=65-140°C; il caso esterno problematico è a T=25°C, temperatura non coperta da AIRES. Aggiungendo AIRES al training, il recall Crystalline sul benchmark esterno è infatti salito da 24% a 48% — un miglioramento reale, anche se non completo.

3. **Bug di parsing dei sali metallici** (`src/chem.py::parse_salt`) — CORRETTO in questa sessione. Sali come ZnBr₂, Zn(acac)₂, ZnSO₄, ZnI₂, Zn(ClO₄)₂ ottenevano `Oxidation_State = NaN` invece di `2`, per una lista di pattern regex incompleta (copriva solo nitrato, cloruro, acetato, metossido). Fix verificato su tutti i sali del progetto, **70/70 test esistenti del progetto continuano a passare**. Non è risultata la causa principale del collasso osservato, ma è un miglioramento di qualità dei dati genuino e a costo zero, mantenuto in `src/chem.py`.

## 4. Cosa è stato effettivamente modificato nel repository

- **`src/chem.py`**: `parse_salt()` corretta (vedi punto 3). Retrocompatibile, nessuna riga di training esistente cambia significato per i sali già ben coperti (nitrati, cloruri).
- **Nessun modello di produzione sostituito.** Gli artefatti sperimentali (`MOF_Experimental_v10_14_AIRES.joblib`, report v10.15/v10.16/v10.17) restano salvati separatamente per riferimento futuro, ma **non sono in uso**.

## 5. Raccomandazione per il prossimo passo (non generico)

Non "più dati" in senso generico — la lacuna è specifica e ora nota con precisione:

> **Sintesi di tipo ZIF/imidazolato con forte eccesso di legante (rapporto >4:1) a temperatura ambiente/bassa (~25-60°C).**

Questa è una condizione sperimentale comune e ben documentata in letteratura (sintesi ZIF-8 a temperatura ambiente con eccesso di 2-metilimidazolo), ma sotto-rappresentata sia nel dataset originale (max 4:1) sia in AIRES (min 65°C). Una ricerca mirata di letteratura con questo identikit preciso avrebbe una probabilità di successo più alta di un'ulteriore espansione generica.

## 6. Metriche di riferimento (per confronto futuro)

**v10.12 (produzione attuale)** — benchmark esterno bloccato v12 (n=33) / laboratorio (n=17):
- Accuracy: 88% / 65%
- Recall Crystalline: 100% / 100%
- Recall Failed: 0% / 0%
- Recall Amorphous: 0% / 0%

Qualsiasi candidato futuro va confrontato su **tutte e quattro** queste cifre insieme, non su una sola isolata — è l'errore metodologico commesso (e corretto) in questa stessa sessione.
