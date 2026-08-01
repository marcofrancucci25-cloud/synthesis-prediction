# Scientific update v10.4.0

This release improves the successful-synthesis layer without altering the frozen three-class outcome predictor.

## Changes

- Rebuilt the positive library from 694 curated crystalline syntheses.
- Parsed precursor hydration, counterion and oxidation-state fields consistently.
- Added completeness, source-quality, diversity and final evidence weights.
- Trained a conditional nearest-neighbour recommendation model using ligand-name character features, metal/linker family, precursor class, solvent, additive, procedure and numerical conditions.
- Added grouped retrospective retrieval validation.
- Retained a fallback heuristic so partial deployments do not crash.

## Important scope statement

No synthetic or unverified literature rows were generated. The positive-support score is not an absolute probability of synthesis success. It quantifies similarity to quality-weighted crystalline precedents and is combined with the independently trained balanced outcome predictor.
