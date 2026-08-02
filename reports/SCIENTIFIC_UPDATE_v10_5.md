# Scientific update v10.5.0

This release implements the requested structural-representation, leakage-audit and label-governance work without overstating model validity.

The ligand registry resolves 58 of 88 unique legacy ligand labels and covers 984 of 1,078 rows. After condition-conflict and contradiction review gates, 661 structure-resolved rows remain eligible for the experimental structural training campaign.

The Morgan/ECFP + RDKit descriptor model achieved mean Macro-F1 0.459 under unseen-ligand grouped validation, but only 0.326 under unseen-scaffold validation. Its scaffold-held-out MCC was 0.064. These results show that molecular fingerprints alone do not cure the dataset's family imbalance and label/provenance limitations.

Accordingly, the structural artifact is included for reproducible research but is not used for production predictions. The frozen externally tested v8 ensemble remains active. Promotion requires source-level DOI/PXRD recuration, resolution of condition/outcome conflicts, and a new frozen external validation.
