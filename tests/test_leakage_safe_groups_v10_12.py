from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_protected_identities_never_cross_connected_groups():
    data = pd.read_csv(ROOT / "data/v12_training_candidates_grouped.csv").fillna("")
    ligand_identity = data.Ligand_InChIKey.where(
        data.Ligand_InChIKey.ne(""), data.Legante.str.casefold()
    )
    for values in [data.Source_DOI, ligand_identity, data.Metal_Ligand_Group]:
        check = pd.DataFrame({"value": values, "group": data.Leakage_Safe_Group})
        check = check[check.value.ne("")]
        assert check.groupby("value").group.nunique().max() == 1
