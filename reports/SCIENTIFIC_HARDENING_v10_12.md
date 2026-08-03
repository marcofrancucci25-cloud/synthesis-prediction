# Scientific hardening v10.12.0

## Production predictor

The production artifact is now `MOF_Audited_Deleaked_v10_12.joblib`. It is
trained on the 731 historical records that passed the deterministic quality
audit. All 347 REVIEW rows are excluded. Solvent volume and its missingness are
absent from the model feature schema; volume remains available only for
physical concentration and validity checks.

Outputs are presented as relative historical-evidence scores. The interface
abstains from numerical interpretation when applicability or class separation
is insufficient. This is intentionally more conservative than the previous
v8 display.

## Chemical identity

Aliases are exact-match only. Positional BDC isomers remain distinct and a
mixed-linker string is never collapsed to one component. Structural identifiers
(SMILES/InChIKey) remain the preferred identity route whenever available.

## Optimizer

If at least three positive templates exist for the exact ligand-metal pair,
candidate generation and positive-support scoring use that exact pair only.
Same-family and same-metal transfer are explicit fallbacks. Applicability now
includes nearest joint-condition support and can no longer reach 1.0 merely
because every marginal category occurred somewhere in the database.

## Evidence acquisition and validation

The release includes a DOI/protocol intake schema for independent failed and
amorphous syntheses. No publications or outcomes are fabricated. All direct
laboratory experiments are locked into a future external campaign, creating a
multiclass benchmark alongside the positive DOI benchmark. The promotion gate
remains closed until every benchmark class has adequate support and the future
training pool reaches independent-DOI and metal-linker diversity thresholds.

A preregistered prospective comparison template is included. It requires
triplicate, randomized, score-blinded comparison of current conditions, a 1:1
DMF baseline, the closest literature protocol and the top three identity-first
model proposals. Only real PXRD/yield results can complete this step.
