from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_lab_quality_gate_counts():
    lab=pd.read_csv(ROOT/'data/laboratory_syntheses_normalized_v10_6.csv')
    assert len(lab)==20
    assert lab.Training_Status.value_counts().to_dict()=={'INCLUDE':17,'EXCLUDE_SPECIAL':2,'REVIEW':1}
    assert lab[lab.Training_Status.eq('INCLUDE')].Outcome_code.value_counts().to_dict()=={2:13,0:3,1:1}

def test_integrated_database_preserves_frozen_training_set():
    frozen=pd.read_csv(ROOT/'data/knowledge_database.csv')
    integrated=pd.read_csv(ROOT/'data/knowledge_database_integrated_v10_6.csv')
    assert len(frozen)==1078
    assert len(integrated)==1095
    assert integrated.ID.astype(str).is_unique

def test_positive_templates_are_unique():
    positive=pd.read_csv(ROOT/'data/successful_synthesis_library_v10_6.csv')
    assert len(positive)==656
    assert positive.Condition_Signature.astype(str).is_unique
    assert (positive.Positive_Library_Source.eq('laboratory_pxrd_positive')).sum()==10

def test_method_aware_condition_groups_have_no_outcome_conflicts():
    lab=pd.read_csv(ROOT/'data/laboratory_syntheses_normalized_v10_6.csv')
    eligible=lab[lab.Training_Status.eq('INCLUDE')]
    assert eligible.groupby('Condition_Group_ID').Outcome_code.nunique().max()==1
    amorphous=eligible[eligible.Sample_ID.eq('DDS4,2')].iloc[0]
    crystalline=eligible[eligible.Sample_ID.eq('DDS4,3')].iloc[0]
    assert amorphous.Microwave_Power_W != crystalline.Microwave_Power_W
