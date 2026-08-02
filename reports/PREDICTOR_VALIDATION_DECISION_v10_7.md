# Predictor validation and promotion decision — v10.7.0

## Decision

The frozen v8.0 ensemble remains the production classifier. Three successor strategies improved recall on positive literature cases but did not preserve adequate three-class discrimination. They were therefore rejected as global replacements.

Version 10.7.0 deploys two safe improvements instead:

1. canonical translation of UI ligand families and common linker aliases before prediction;
2. a separate verified-evidence layer containing 17 eligible laboratory experiments and 9 PXRD/XRD-supported literature protocols.

Experimental evidence is displayed alongside, but never blended into, the calibrated model probabilities.

## Validation results

| Strategy | Legacy grouped balanced accuracy | Macro F1 | MCC | Laboratory crystalline recall | Literature crystalline recall | Decision |
|---|---:|---:|---:|---:|---:|---|
| Reconstructed v8 schema | 0.825 | 0.822 | 0.721 | 0/13 on held-out laboratory set | 8/9 | Production baseline |
| Canonical, volume removed | 0.581 | 0.582 | 0.460 | 2/13 | 9/9 | Rejected |
| Canonical, volume retained but missing-volume flag removed | 0.681 | 0.692 | 0.564 | 6/13 | 9/9 | Rejected |
| Lab-augmented, condition-group holdout | 0.666 | 0.680 | 0.556 | 12/13 | 9/9 | Rejected globally |

The lab-augmented candidate learned the crystalline laboratory domain, but recognized only 1/3 failed experiments and 0/1 amorphous experiment in condition-group holdout. Its apparent 76.5% laboratory accuracy is therefore misleading; laboratory balanced accuracy was only 0.419.

## Production effect of canonical input handling

On the nine-protocol literature challenge, canonical input handling retains 8/9 crystalline argmax predictions, raises the number with `P(crystalline) ≥ 0.5` from 6/9 to 7/9, and raises mean `P(crystalline)` from 0.608 to 0.635.

For the aqueous UiO-66 protocol from DOI `10.1002/adsu.202500854`, `P(crystalline)` increases from 0.451 to 0.492 and crystalline remains the argmax class. The verified-evidence layer also returns the exact PXRD-supported protocol and its DOI.

For UiO-66 from ZrCl4/DMF/H2O, the classifier remains uncertain and does not predict crystalline. The exact literature precedent is therefore displayed as crystalline evidence with interpretive priority, while the model probability remains visible and unchanged.

## Scientific boundaries

- The nine literature records are positive-only; they do not estimate specificity.
- Training-source DOI provenance is incomplete, so literature overlap with historical records cannot be excluded completely.
- Only one eligible amorphous laboratory record is available; its class-level estimate is highly uncertain.
- Multistage protocols, pH, addition order and quantitative modulator concentration remain incompletely represented by the v8 feature schema.
