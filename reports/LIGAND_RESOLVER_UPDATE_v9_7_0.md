# Ligand Resolver Update v9.7.0

- Conservative query expansion and abbreviation extraction.
- PubChem exact/word search with multiple CID candidates and synonyms.
- RDKit parent, charge and tautomer standardization before deduplication.
- Multi-candidate formula search.
- Portable confirmed-ligand cache (`data/confirmed_ligands.json`).
- Tavily fallback only for alternate textual identifiers, followed by chemical-database validation.
- External test and predictive model unchanged.
