# Prediction Validity Gate v10.4.1

This release addresses the principal issue identified in the local validation campaign: tree-based probabilities could remain high or increase for experimental conditions outside the training range.

## Changes
- Added empirical range checks for temperature, time, L:M ratio, reagent amounts and solvent volume.
- Added concentration plausibility checks.
- Added consistency control between entered reagent amounts and the declared L:M ratio.
- Added an explicit `Outside validated experimental range` state.
- Integrated numerical support into the applicability-domain score.
- Probabilities remain visible for transparency, but the interface explicitly states that they are not reliable outside the validated range.

The gate does not fabricate corrected probabilities and does not modify the frozen predictive model.
