# App v10.10.0 — v11 literature expansion tranche 02

## Outcome

The production predictor remains the frozen v8 ensemble. This release expands the auditable v11 development set with a complete high-throughput Al-PMOF campaign, but it does not promote a new model before the independent-source and class-balance gates are met.

## Verified Al-PMOF campaign

Forty-five microwave experiments were transcribed from Tables S3–S5 of the supporting information for DOI `10.1038/s42004-022-00785-2`. Every record includes:

- AlCl3·6H2O and H2TCPP quantities resolved from the reported concentration code;
- 80:20 water/organic-solvent volumes;
- microwave power, temperature and time;
- the authors' original PXRD categorical score;
- the reported yield for generation 2;
- article DOI and characterization-data DOI (`10.5281/zenodo.7186602`).

The outcome conversion follows the article's own scale:

- score 1 → failed/no isolated powder (class 0);
- scores 2–5 → amorphous or poorly crystalline (class 1);
- scores 6–10 → crystalline Al-PMOF (class 2).

This adds 4 failed, 16 amorphous/poorly crystalline and 25 crystalline records. Yield is retained as a separate field and is never used to override the PXRD label.

## Anti-dominance weighting

All 45 experimental conditions remain available as distinct evidence. For future fitting, however, one article is capped at 20 condition-equivalents. This prevents a single robotic campaign from numerically dominating independent sources while preserving within-study condition variation.

## Current gold-set readiness

- 79 total gold records;
- 63 training candidates and 16 locked external records;
- raw training distribution: 7 failed, 17 amorphous/poorly crystalline, 39 crystalline;
- weighted effective distribution: 4.778, 7.611 and 22.611 condition-equivalents;
- 2 independent training DOI groups;
- 4 metal–linker groups;
- 2 ligand families;
- zero DOI overlap between training and the locked benchmark.

The v11 promotion gate therefore remains closed. The next tranche must prioritize failed and amorphous outcomes from new DOI groups and additional ligand families, rather than adding further crystalline variants from the same sources.

## User-facing literature match

The exact Al(III)–TCPP pair now resolves to Al-PMOF and the verified article DOI in the literature-candidate panel. It remains explicitly presented as a literature precedent, not as identification of the user's isolated phase.
