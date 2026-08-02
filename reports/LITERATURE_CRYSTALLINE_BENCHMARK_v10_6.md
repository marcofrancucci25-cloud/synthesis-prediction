# Benchmark locale su sintesi cristalline di letteratura — v10.6

## Scopo

Questo controllo misura la **recall della classe cristallina** del predittore v8 congelato su nove protocolli pubblicati e caratterizzati mediante PXRD/XRD. Non è una validazione esterna completa: il set contiene soltanto esiti cristallini, quindi non misura specificità, falsi positivi o balanced accuracy. Inoltre, il database storico non conserva i DOI delle fonti e non consente di escludere ogni sovrapposizione con la letteratura di addestramento.

## Risultati

| Caso | P(fallito) | P(amorfo) | P(cristallino) | Classe prevista | Dominio |
|---|---:|---:|---:|---|---|
| UiO-66 acquoso, 4.0 mM NaOH | 32.1% | 22.8% | 45.1% | Cristallino | Intermedio |
| UiO-66, ZrCl4/DMF/H2O | 53.1% | 25.7% | 21.3% | **Fallito** | Intermedio |
| ZIF-8 | 12.2% | 21.0% | 66.8% | Cristallino | Intermedio |
| ZIF-67 | 20.8% | 18.7% | 60.5% | Cristallino | Intermedio |
| HKUST-1 | 47.0% | 5.9% | 47.1% | Cristallino | Intermedio |
| Mg-MOF-74 | 14.8% | 6.0% | 79.2% | Cristallino | Interno |
| MIL-101(Cr) | 9.0% | 26.9% | 64.1% | Cristallino | Intermedio |
| MIL-53(Al) | 8.2% | 4.0% | 87.8% | Cristallino | Fuori dominio |
| MOF-5 | 21.1% | 3.9% | 75.1% | Cristallino | Fuori dominio |

- Recall positiva per argmax: **8/9 = 88.9%**.
- Casi con P(cristallino) almeno 50%: **6/9**.
- Falso negativo: **UiO-66 da ZrCl4/DMF/H2O**, verificato cristallino in letteratura ma classificato come fallito.
- HKUST-1 è formalmente corretto ma instabile: 47.1% cristallino contro 47.0% fallito.
- Il caso UiO-66 acquoso segnalato dall'utente resta poco convincente: la classe cristallina vince, ma soltanto con 45.1%. Piccole differenze di volume, nome del legante o codifica dell'additivo possono riportarlo nell'intervallo ~35–45%.

## Problemi scoperti

### 1. Incompatibilità completa delle categorie di famiglia

Le categorie mostrate dall'interfaccia (`Carboxylate`, `Imidazolate/azolate`, ecc.) non coincidono con nessuna categoria usata dal modello (`Carbossilati aromatici`, `Imidazolati`, `Bipyrazole`, ecc.). L'intersezione è vuota. Di conseguenza, l'encoder tratta sempre la famiglia inserita dall'utente come sconosciuta.

Una prova in memoria con categorie e nomi compatibili modifica P(cristallino) fino a **+14.4 punti percentuali** (HKUST-1). La correzione non risolve però da sola il falso negativo UiO-66.

### 2. Missingness leakage nel volume del solvente

Nel training, `Volume solvente` è mancante nel **73.4%** dei record. La mancanza è fortemente associata all'etichetta:

| Classe | Volume mancante |
|---|---:|
| Fallito | 3.2% |
| Amorfo/incerto | 86.3% |
| Cristallino | 93.7% |

L'interfaccia richiede invece sempre un volume numerico. Nel challenge, sostituire artificialmente il volume reale con un valore mancante aumenta P(cristallino) in media di **23.2 punti percentuali**; per UiO-66 ZrCl4/DMF/H2O l'aumento è **45.9 punti**, trasformando la previsione da fallito a cristallino. Questo indica che il modello ha appreso anche la modalità di compilazione dei dati, non soltanto la chimica.

### 3. Rappresentazione insufficiente dei protocolli

Il modello accetta una sola temperatura, un solo tempo, un solvente testuale e un additivo testuale. Non può descrivere correttamente:

- pre-formazione del cluster di zirconio;
- pH e concentrazione di NaOH;
- sequenze di riscaldamento multistadio;
- ordine di aggiunta, agitazione, aging e pressione;
- volumi separati delle soluzioni e composizione quantitativa delle miscele.

Il protocollo acquoso UiO-66 dipende proprio da pre-formazione del cluster e controllo del pH; MOF-5 usa una sequenza termica a due stadi. La compressione in un'unica riga perde quindi variabili causali importanti.

## Priorità raccomandate

1. **Correggere subito lo schema di input**, traducendo le categorie dell'interfaccia nelle categorie storiche oppure, preferibilmente, ricostruendo un vocabolario canonico unico prima di ogni nuovo training.
2. **Non riaddestrare usando il volume così com'è**. Prima occorre recuperare i volumi mancanti quando possibile, aggiungere un indicatore di provenienza/missingness e verificare che la mancanza non predica l'esito.
3. **Aggiungere campi strutturati** per pH, modulatore, concentrazione del modulatore, composizione dei solventi, numero di stadi, aging e metodo di riscaldamento.
4. **Creare un vero test esterno congelato**, con DOI, PXRD, protocolli completi, esempi positivi e negativi e split per scaffold/legante non visto.
5. **Mostrare separatamente evidenza e modello**: se esiste un precedente letterario esatto cristallino, l'interfaccia deve segnalarlo anche quando il classificatore assegna una probabilità bassa.

## Fonti primarie del challenge

- UiO-66 acquoso: https://doi.org/10.1002/adsu.202500854
- UiO-66 ZrCl4/DMF/H2O: https://doi.org/10.1021/acs.inorgchem.0c00991
- ZIF-8: https://doi.org/10.1039/C3CE42485E
- ZIF-67: https://doi.org/10.1021/acsami.4c07877
- HKUST-1: https://doi.org/10.1016/j.tca.2016.11.013
- Mg-MOF-74: https://doi.org/10.1021/acsomega.0c01189
- MIL-101(Cr): https://doi.org/10.1016/j.heliyon.2024.e31341
- MIL-53(Al): https://doi.org/10.1021/jacs.9b07557
- MOF-5: https://doi.org/10.1021/acsomega.8b02332
