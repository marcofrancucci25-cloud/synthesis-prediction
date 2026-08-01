# Ligand Resolver Update v9.6.0

This release introduces a precision-first consensus resolver.

- Curated MOF linker entries retain highest priority.
- Direct SMILES are parsed locally with RDKit.
- Systematic names are queried through the official EMBL-EBI OPSIN service.
- PubChem PUG REST and NCI Cactus provide independent candidates.
- Candidates are canonicalized and deduplicated by InChIKey/connectivity.
- Automatic acceptance requires independent source agreement or a unique strong OPSIN result.
- Conflicting or insufficiently corroborated candidates require explicit user confirmation.
- Molecular-formula searches always require confirmation because formulas are not structurally unique.
