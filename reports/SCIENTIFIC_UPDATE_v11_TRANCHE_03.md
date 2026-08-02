# App v10.11.0 — v11 literature expansion tranche 03

## Outcome

This release adds 100 primary-literature protocols to the auditable v11 development dataset. The production predictor remains the frozen v8 ensemble: the promotion gate is deliberately closed until class balance, independent-source coverage and metal–linker diversity all pass together.

## HKUST-1 robotic campaign

All 90 experimental rows from Supplementary Data 1 of DOI `10.1038/s41467-019-08483-9` were imported from the authors' machine-readable CSV. The dataset preserves:

- water, DMF, ethanol, methanol and isopropanol volumes;
- published metal/BTC ratio, converted transparently to the app's ligand/metal convention;
- temperature, microwave power and time;
- the continuous normalized PXRD crystallinity/phase-purity fitness score;
- BET as a separate raw field.

The article does not provide three-class labels for these rows. The following policy is therefore an explicit app curation rule, not an author label:

- score ≤ 0.30 → class 0, very-low target crystallinity / failed target synthesis;
- score 0.35–0.75 → class 1, partially successful / poor target crystallinity;
- score ≥ 0.80 → class 2, high-crystallinity target MOF.

This produces 6 class-0, 48 class-1 and 36 class-2 records. The original continuous score and policy identifier `V11-PXRD-NORMALIZED-0.30-0.80-v1` remain attached to every row. BET is not used in the mapping because a zero can also denote an unmeasured sample. Absolute precursor amounts and copper-nitrate hydration are not present in the shared CSV and were not invented; an evidence-quality factor of 0.85 records that limitation for future fitting.

Primary sources:

- article DOI: https://doi.org/10.1038/s41467-019-08483-9
- archived data DOI: https://doi.org/10.24435/materialscloud:2018.0011/v3
- official supplementary CSV: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-019-08483-9/MediaObjects/41467_2019_8483_MOESM5_ESM.csv

## MOF-321 and MOF-322 protocols

Tables 1 and 2 of DOI `10.1021/acscentsci.3c01087` explicitly identify ten representative high-crystallinity conditions: five for MOF-321 and five for MOF-322. Metal, ligand, NaOH, water, temperature, microwave time and 300 W power were transcribed directly. These ten class-2 labels are direct article designations and do not use an inferred numerical threshold.

Primary sources:

- article DOI: https://doi.org/10.1021/acscentsci.3c01087
- open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC10683477/
- supporting information: https://pubs.acs.org/doi/suppl/10.1021/acscentsci.3c01087/suppl_file/oc3c01087_si_001.pdf

The literature-candidate registry now resolves only exact Al(III)–H2PZVDC and Al(III)–H2TVDC pairs to MOF-321 and MOF-322 respectively. It does not cross-match those linkers and does not claim that composition alone identifies an isolated phase.

## Current gold-set readiness

- 179 total gold records;
- 163 training candidates and 16 locked external records;
- raw training distribution: 13 class-0, 65 class-1 and 85 class-2;
- weighted effective distribution: 5.911, 16.583 and 39.128 condition-equivalents;
- 4 independent training DOI groups;
- 7 metal–linker groups;
- 5 ligand families;
- zero DOI overlap between training and the locked benchmark.

Only the five-family diversity gate now passes. The next curation tranche must prioritize genuine negative/failed outcomes from new articles and at least three additional metal–linker pairs. A v11 retraining performed now would remain vulnerable to source-specific bias and poor class-0 calibration, so no production model file is changed in this release.
