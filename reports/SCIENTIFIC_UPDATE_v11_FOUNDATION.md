# App v10.9.0 — v11 model-development foundation

## Decision

The production predictor has **not** been retrained. The first v11 step is a provenance-first dataset that separates eligible training candidates from a locked external benchmark before any model fitting occurs.

## Existing-data audit

- The historical model database contains 1,078 labelled records (217 failed, 306 amorphous/uncertain, 555 crystalline), but it does not contain DOI provenance or record-level PXRD/SCXRD evidence.
- Those records remain valid for reproducing the frozen v8 model, but they are not silently promoted to the v11 gold set.
- Seventeen direct laboratory records pass the existing quality gate; DDS1 remains under review and the two in-situ ibuprofen syntheses remain outside the synthesis-outcome domain.

## First verified expansion tranche

- Seven additional ZIF-8 protocols were extracted from the same primary article as the locked nitrate protocol. They use Zn(acac)2, ZnSO4, Zn(ClO4)2, Zn(OAc)2, ZnCl2, ZnI2 and ZnBr2 under the reported Zn/Hmim/MeOH ratio of 1/8/559 for 1 h at room temperature. The article reports pure-phase ZIF-8 by PXRD for every listed precursor.
- One complete Co(BPZNH2)·DMF protocol was transcribed from the primary experimental section: 1.00 mmol ligand, 1.00 mmol Co(NO3)2·6H2O, 15 mL DMF, 393 K, 24 h, with PXRD whole-pattern characterization.

All ZIF-8 variants from DOI `10.1039/C3CE42485E` remain in the locked external partition. Even though their precursors differ, placing records from one article in both training and test would create source leakage.

## Current v11 gold-set status

- 34 total gold records;
- 18 training candidates: 3 failed, 1 amorphous/uncertain, 14 crystalline;
- 16 locked external records, including all eight protocols from the ZIF-8 zinc-salt study;
- no DOI overlap between training candidates and the locked benchmark.

Repeated laboratory batches are retained as evidence, but exact repeated-condition signatures receive reciprocal training weights so that their combined influence is never greater than one independent condition.

## Promotion gate

Retraining remains blocked until the training pool has at least:

- 30 records in each outcome class;
- 20 independent literature DOI groups;
- 10 metal–linker groups;
- 5 ligand families;
- zero DOI overlap with external evaluation.

The immediate curation priority is therefore failed and amorphous/uncertain syntheses from independent sources, followed by broader linker-family coverage.
