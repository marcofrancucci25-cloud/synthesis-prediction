# Negative and amorphous literature expansion — v10.13

## Scope and result

- 759 DOI-linked negative/amorphous records were added.
- Class 0: 673 records explicitly labelled `no solid product` by the authors.
- Class 1: 85 records explicitly labelled `no crystalline product` in the
  high-throughput campaign key, plus one direct aqueous aZIF synthesis whose
  amorphous assignment was supported by XRD and SAED.
- 71 `unknown phase` records were isolated in a review file and were not mapped
  to a model outcome.
- No missing outcome, unpublished silence, or unmatched reference pattern was
  converted into a negative label.

## Sources

1. Gaidimas et al., *Mapping the crystallization landscape of rare earth MOFs:
   a high-throughput investigation of structure, kinetics, and selectivity*.
   Article DOI: `10.1039/d5sc09992g`. Public campaign-data DOI:
   `10.5281/zenodo.17902549`.
2. Wu et al., *Packaging and delivering enzymes by amorphous metal-organic
   frameworks*. DOI: `10.1038/s41467-019-13153-x`.

## Files

- `data/v13_source_rare_earth_landscape.csv`: verbatim 1,488-row campaign key.
- `data/v13_negative_amorphous_literature.csv`: curated negative/amorphous
  evidence with a DOI on every row.
- `data/v13_literature_records_needing_review.csv`: 71 unknown-phase cases.
- `tools/curate_v13_negative_literature.py`: reproducible mapping and integrity
  checks.

## Model decision

The production model was not retrained. Although the tranche materially expands
the evidence database, 758/759 records originate from one article and the
campaign key omits absolute precursor amounts, ligand:metal ratio, and solvent
volume. Treating these correlated, partially specified observations as 758
independent full protocols would inflate apparent sample size and undermine the
leakage-safe validation policy. They are instead exposed through the verified
precedent layer; the v10.12 audited model remains frozen.
