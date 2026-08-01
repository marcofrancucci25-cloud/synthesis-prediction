# Scientific update v10.0 — Separate prediction and joint optimization

## Prediction Engine
Evaluates the exact conditions entered by the user and returns the calibrated three-class outcome probabilities. No input is changed during prediction.

## Joint Optimization Engine
Keeps only ligand identity and metal identity fixed. It jointly samples and ranks every variable present in the frozen v8.0 feature schema: precursor/counterion, oxidation state, hydration, solvent, additive, temperature, time, reagent amounts, ligand-to-metal ratio, and solvent volume.

The search combines mixed categorical and continuous sampling, feasibility filtering, applicability-domain penalties, user constraints, multi-objective scoring, and Pareto ranking.

## Scientific boundary
pH, solvent fractions, modulator equivalents, heating ramp, cooling rate, stirring, addition order, vessel filling fraction and synthesis method are represented in the new v10 dataset schema but are not optimized by the frozen model because they were not consistently available during training. They must be curated and followed by model retraining and a new grouped external validation before being treated as learned optimization variables.
