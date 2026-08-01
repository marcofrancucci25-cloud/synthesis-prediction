# Scientific update v10.3

The optimization architecture now has two independent evidence layers.

1. **Balanced outcome predictor** — estimates the probabilities of failure, amorphous/uncertain product, and crystalline MOF from the exact condition vector.
2. **Successful-synthesis recommendation layer** — learns realistic joint condition patterns from positive crystalline records and supplies template generation and precedent support.

The positive layer is deliberately not described as a success-probability model because published positive syntheses are affected by publication bias and cannot quantify failure risk without negative data.

Candidate generation uses approximately 72% successful-template mutations and 28% broad exploration. Every generated candidate is subsequently evaluated by the frozen three-class predictor, feasibility rules, applicability-domain checks, and positive-precedent similarity. Pareto ranking preserves alternatives across probability, precedent, domain, feasibility, resource use, and distance from the starting experiment.

The bundled successful-synthesis library contains 694 deduplicated positive records, 92 unique ligands, and 20 metals. It is an initial project-derived knowledge layer, not yet a comprehensive corpus of all published MOF syntheses.
