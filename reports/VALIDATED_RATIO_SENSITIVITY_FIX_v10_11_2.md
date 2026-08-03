# Validated L:M sensitivity correction — v10.11.2

## Defect corrected

The previous local sensitivity grid generated `0.1` from a current L:M ratio of `1.0` by applying a fixed `-1.0` perturbation and clipping the result. It then changed only `Rapporto_LM`, leaving ligand and metal amounts unchanged. A nominal ratio of `0.1` could therefore coexist with 0.1 mmol ligand and 0.1 mmol metal, which physically corresponds to 1:1.

## New behaviour

For a candidate ratio \(r=L/M\) and current total precursor amount \(T=L+M\), the sensitivity layer now calculates:

\[
M = \frac{T}{1+r}, \qquad L = \frac{rT}{1+r}.
\]

This preserves total precursor concentration when solvent volume is unchanged and guarantees that the explicit ratio agrees with both mmol fields.

Candidate ratios are exact values observed in the frozen training database. Selection proceeds from the same ligand–metal system to the same metal/family, then the same metal, with the global dataset only as a fallback. Values outside the central 90% of the training distribution are not used for local sensitivity. Every complete perturbed record must also pass `prediction_validity()` before evaluation.

## Scientific boundary

The sensitivity panel remains model-based and non-causal. It reports how the frozen classifier responds to a controlled, coherent perturbation; it is not an experimental synthesis recommendation. Complete proposals remain the responsibility of the separate joint optimizer, which combines probability, positive precedents, feasibility and applicability.
