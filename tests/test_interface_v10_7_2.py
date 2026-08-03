from pathlib import Path

from src.chem import hsab_acid_class


def test_hsab_classification_uses_oxidation_state():
    assert hsab_acid_class('Cu',1)=='Soft acid'
    assert hsab_acid_class('Cu',2)=='Borderline acid'
    assert hsab_acid_class('Fe',2)=='Borderline acid'
    assert hsab_acid_class('Fe',3)=='Hard acid'
    assert hsab_acid_class('Zr',4)=='Hard acid'
    assert hsab_acid_class('Ag',1)=='Soft acid'


def test_diagnostic_expanders_are_hidden_and_optimizer_reset_is_present():
    source=Path('app.py').read_text(encoding='utf-8')
    assert 'st.expander("Scientific scope of this optimization")' not in source
    assert 'st.expander("Similar experimental records")' not in source
    assert '"Reset optimizer fields"' in source
    assert 'on_click=_reset_optimizer_inputs' in source


def test_release_version_includes_interface_fix():
    assert Path('VERSION').read_text(encoding='utf-8').strip()=='10.11.8'
