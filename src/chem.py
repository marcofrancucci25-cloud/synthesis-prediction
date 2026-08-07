import re, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METALS=json.loads((ROOT/'models/metal_properties.json').read_text())

COMMON_LIGAND_ALIASES={
 'h2bdc':'terephthalic acid','bdc':'terephthalic acid','btc':'benzene-1,3,5-tricarboxylic acid',
 'h3btc':'benzene-1,3,5-tricarboxylic acid','bpdc':'biphenyl-4,4-dicarboxylic acid','dobdc':'2,5-dihydroxyterephthalic acid',
 'bpy':'2,2-bipyridine','4,4-bpy':'4,4-bipyridine','bpz':'4,4-bipyrazole','h2bpz':'4,4-bipyrazole',
 'imidazole':'imidazole','2-methylimidazole':'2-methylimidazole','hmim':'2-methylimidazole','trimesic acid':'benzene-1,3,5-tricarboxylic acid'
}
FAMILIES=['Bipyrazole/pyrazole','Carboxylate','Imidazolate/azolate','Pyridyl/N-donor','Phosphonate','Sulfonate','Curcumin/β-diketonate','Mixed donor','Other/unknown']
COUNTERIONS=['nitrate','acetate','chloride','bromide','iodide','sulfate','perchlorate','triflate','tetrafluoroborate','hexafluorophosphate','carbonate','hydroxide','oxide','alkoxide','acetylacetonate','other']

HARD_ACIDS={
 'Li','Be','Mg','Ca','Sr','Ba','Al','Sc','Y','La','Ce','Pr','Nd','Sm','Eu',
 'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Ga','In','Zr','Hf','Th','U',
}
SOFT_ACIDS={'Ag','Au','Cd','Hg','Pd','Pt'}

def hsab_acid_class(metal, oxidation_state=2):
    """Return an indicative HSAB class for a metal ion.

    HSAB character belongs to the ion, not only to the element.  The selected
    oxidation state is therefore used for common variable-valence metals.
    """
    symbol=str(metal or '').strip()
    try:
        oxidation=int(oxidation_state)
    except (TypeError,ValueError):
        oxidation=None
    if symbol in HARD_ACIDS:
        return 'Hard acid'
    if symbol in SOFT_ACIDS:
        return 'Soft acid'
    if symbol=='Cu':
        return 'Soft acid' if oxidation==1 else ('Borderline acid' if oxidation else 'Oxidation-state dependent')
    if symbol in {'Fe','Co','Cr','Mn'}:
        return 'Hard acid' if oxidation is not None and oxidation>=3 else ('Borderline acid' if oxidation else 'Oxidation-state dependent')
    if symbol in {'Ti','V','Nb','Ta'}:
        return 'Hard acid' if oxidation is not None and oxidation>=4 else ('Borderline acid' if oxidation else 'Oxidation-state dependent')
    if symbol in {'Mo','W'}:
        return 'Hard acid' if oxidation is not None and oxidation>=5 else ('Borderline acid' if oxidation else 'Oxidation-state dependent')
    if symbol=='Re':
        return 'Hard acid' if oxidation is not None and oxidation>=6 else ('Borderline acid' if oxidation else 'Oxidation-state dependent')
    if symbol=='Sn':
        return 'Hard acid' if oxidation is not None and oxidation>=4 else ('Borderline acid' if oxidation else 'Oxidation-state dependent')
    if symbol=='Pb':
        return 'Hard acid' if oxidation is not None and oxidation>=4 else 'Borderline acid'
    return 'Borderline acid'

# The v8 training data use an older Italian family vocabulary.  The public UI
# intentionally exposes clearer English labels, so values must be translated
# before they enter either the frozen model or a retrained successor.
TRAINING_FAMILIES={
 'Altro','Bipyrazole','Carbossilati alifatici','Carbossilati aromatici','Curcumina',
 'Imidazolati','Non specificata','Organometallica','Organometallica - pirazoli',
 'Organometallica - scambio anionico','Pyrazole carbossilati','Triazole'
}

def canonicalize_family(family, ligand=''):
    raw=str(family or '').strip()
    if raw in TRAINING_FAMILIES:
        return raw
    low=raw.casefold(); lig=str(ligand or '').casefold()
    if low=='carboxylate':
        aliphatic=('fumar' ,'succin', 'malonic', 'oxalic', 'adipic', 'glutaric', 'maleic')
        return 'Carbossilati alifatici' if any(k in lig for k in aliphatic) else 'Carbossilati aromatici'
    if low=='imidazolate/azolate':
        return 'Triazole' if 'triazol' in lig else 'Imidazolati'
    if low=='bipyrazole/pyrazole':
        return 'Pyrazole carbossilati' if any(k in lig for k in ('carbox','benzoic','bdc')) else 'Bipyrazole'
    if low=='curcumin/β-diketonate':
        return 'Curcumina'
    if low in {'pyridyl/n-donor','phosphonate','sulfonate','mixed donor','other/unknown'}:
        return 'Non specificata'
    return raw or 'Non specificata'

MODEL_LIGAND_ALIASES={
 'h2bdc':'1,4-Benzenedicarboxylic acid (H2BDC)',
 'bdc':'1,4-Benzenedicarboxylic acid (H2BDC)',
 'terephthalic acid':'1,4-Benzenedicarboxylic acid (H2BDC)',
 'benzene-1,4-dicarboxylic acid':'1,4-Benzenedicarboxylic acid (H2BDC)',
 '1,4-benzenedicarboxylic acid':'1,4-Benzenedicarboxylic acid (H2BDC)',
 'h3btc':'1,3,5-Benzenetricarboxylic acid (H3BTC)',
 'btc':'1,3,5-Benzenetricarboxylic acid (H3BTC)',
 'trimesic acid':'1,3,5-Benzenetricarboxylic acid (H3BTC)',
 'benzene-1,3,5-tricarboxylic acid':'1,3,5-Benzenetricarboxylic acid (H3BTC)',
 'dobdc':'2,5-Dihydroxyterephthalic acid (H4DOBDC)',
 'h4dobdc':'2,5-Dihydroxyterephthalic acid (H4DOBDC)',
 '2,5-dihydroxyterephthalic acid':'2,5-Dihydroxyterephthalic acid (H4DOBDC)',
 '1,3-bdc':'Isophthalic acid (1,3-BDC)',
 'isophthalic acid':'Isophthalic acid (1,3-BDC)',
 'benzene-1,3-dicarboxylic acid':'Isophthalic acid (1,3-BDC)',
 '1,3-benzenedicarboxylic acid':'Isophthalic acid (1,3-BDC)',
 '1,2-bdc':'Phthalic acid (1,2-BDC)',
 'phthalic acid':'Phthalic acid (1,2-BDC)',
 'benzene-1,2-dicarboxylic acid':'Phthalic acid (1,2-BDC)',
 '1,2-benzenedicarboxylic acid':'Phthalic acid (1,2-BDC)',
 'hmim':'2-Methylimidazole',
 '2-methylimidazole':'2-Methylimidazole',
}

def canonicalize_ligand_for_model(text):
    """Return a stable model-facing name for common linker aliases.

    Resolver output may contain ``alias | canonical name``.  Matching each side
    prevents harmless naming differences from looking like an unseen ligand to
    the character n-gram model.
    """
    raw=' '.join(str(text or '').strip().split())
    # A slash denotes a genuine mixed-linker formulation in the historical
    # database (for example 2,6-NDC / H3BTC).  Never collapse the whole system
    # to one component merely because an alias occurs as a substring.
    if '/' in raw:
        return raw
    parts=[p.strip().casefold() for p in raw.split('|') if p.strip()]
    for part in parts or [raw.casefold()]:
        if part in MODEL_LIGAND_ALIASES:
            return MODEL_LIGAND_ALIASES[part]
    # Deliberately no substring matching here.  In particular, ``BDC`` inside
    # ``1,3-BDC`` must not turn isophthalic acid into terephthalic acid.
    return raw

def normalize_ligand(text):
    t=' '.join(str(text or '').strip().split())
    return COMMON_LIGAND_ALIASES.get(t.lower(),t)

def infer_family(text):
    s=str(text or '').lower()
    if 'pyraz' in s or 'bpz' in s: return 'Bipyrazole/pyrazole'
    if any(k in s for k in ['carbox','benzoic','bdc','btc','fumar','terephthal','trimesic']): return 'Carboxylate'
    if 'imidazol' in s or 'triazol' in s or 'tetrazol' in s: return 'Imidazolate/azolate'
    if 'pyrid' in s or 'bipyrid' in s or 'phenanthrol' in s: return 'Pyridyl/N-donor'
    if 'phosphon' in s: return 'Phosphonate'
    if 'sulfon' in s: return 'Sulfonate'
    if 'curc' in s or 'diketon' in s: return 'Curcumin/β-diketonate'
    return 'Other/unknown'

_ANION_CHARGE = {
    'nitrate': 1, 'acetate': 1, 'perchlorate': 1, 'bromide': 1, 'chloride': 1,
    'iodide': 1, 'fluoride': 1, 'hydroxide': 1, 'triflate': 1,
    'tetrafluoroborate': 1, 'hexafluorophosphate': 1, 'acetylacetonate': 1,
    'alkoxide': 1, 'sulfate': 2, 'carbonate': 2, 'oxalate': 2,
}


def parse_salt(formula):
    s=str(formula or '').replace(' ', '').replace('•','·')
    hyd=0.0
    m=re.search(r'·([0-9]+(?:\.[0-9]+)?)H2O',s,re.I)
    if m: hyd=float(m.group(1))
    elif 'H2O' in s: hyd=1.0
    sl=s.lower()
    # Anion identification. NOTE: 'iodide' is matched separately (against the
    # ORIGINAL-case string) because a lowercase 'i' is far too common a
    # substring (acetylacetonate, oxide, etc.) to use as a safe token once
    # the formula is lowercased; the capital element symbol 'I' immediately
    # followed by a digit/end-of-string is a much more reliable anchor.
    tests=[('nitrate','no3'),('acetate','oac'),('perchlorate','clo4'),('sulfate','so4'),('triflate','cf3so3'),('tetrafluoroborate','bf4'),('hexafluorophosphate','pf6'),('bromide','br'),('chloride','cl'),('hydroxide','oh'),('alkoxide','och3'),('acetylacetonate','acac')]
    an='other'
    if re.search(r'I(?=[0-9)]|$)', s):
        an='iodide'
    else:
        for name,token in tests:
            if token in sl: an=name; break
    ox=np.nan
    roman={'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6}
    rm=re.search(r'\(([ivx]+)\)',sl)
    if rm:
        ox=roman.get(rm.group(1),np.nan)
    else:
        # Explicit patterns first (kept from the original implementation for
        # backward compatibility / readability on the most common salts).
        for pat,val in [(r'no3\)3',3),(r'cl3',3),(r'no3\)2',2),(r'cl2',2),(r'oac\)2',2),(r'cl4',4),(r'och3\)4',4)]:
            if re.search(pat,sl): ox=val; break
        if np.isnan(ox):
            # General charge-balance fallback: read an explicit multiplier
            # directly after the anion token / its closing parenthesis
            # (covers e.g. ZnBr2, Zn(acac)2, Zn(ClO4)2, ZnI2), otherwise -
            # for anion tokens with a known charge - assume a single anion
            # unit balances the cation (covers e.g. ZnSO4).
            mult=None
            m2=re.search(r'(?:no3|oac|clo4|acac|bf4|pf6)\)?(\d)',sl)
            if m2:
                mult=int(m2.group(1))
            else:
                m3=re.search(r'(?:cl|br)(\d)',sl)
                if m3:
                    mult=int(m3.group(1))
                elif an=='iodide':
                    m4=re.search(r'I(\d)',s)
                    if m4: mult=int(m4.group(1))
            if mult is not None:
                ox=mult*_ANION_CHARGE.get(an,1)
            elif an in _ANION_CHARGE:
                ox=_ANION_CHARGE[an]
    return {'Hydration_Number':hyd,'Counterion_Class':an,'Oxidation_State':ox}

def build_row(values):
    row=dict(values)
    row['Legante']=canonicalize_ligand_for_model(normalize_ligand(row.get('Legante','')))
    public_family=row.get('Famiglia_Legante') or infer_family(row['Legante'])
    row['Famiglia_Legante']=canonicalize_family(public_family,row['Legante'])
    row['Ligand_Text']=(row['Legante']+' '+row['Famiglia_Legante']).lower()
    parsed=parse_salt(row.get('Sale_Metallico',''))
    for k,v in parsed.items():
        if row.get(k) in (None,'', 'auto'): row[k]=v
    vals=METALS.get(str(row.get('Metallo','')), [np.nan,np.nan,np.nan,np.nan,'unknown',np.nan])
    row['Metal_Atomic_Number'],row['Metal_Atomic_Weight'],row['Metal_Group'],row['Metal_Period'],row['Metal_Block'],row['Metal_Electronegativity']=vals
    return pd.DataFrame([row])

def precursor_formula(metal, oxidation, counterion, hydration):
    charges={'nitrate':-1,'acetate':-1,'chloride':-1,'bromide':-1,'iodide':-1,'perchlorate':-1,'triflate':-1,'tetrafluoroborate':-1,'hexafluorophosphate':-1,'sulfate':-2,'carbonate':-2,'hydroxide':-1,'oxide':-2}
    symbols={'nitrate':'NO3','acetate':'OAc','chloride':'Cl','bromide':'Br','iodide':'I','perchlorate':'ClO4','triflate':'OTf','tetrafluoroborate':'BF4','hexafluorophosphate':'PF6','sulfate':'SO4','carbonate':'CO3','hydroxide':'OH','oxide':'O'}
    q=charges.get(counterion,-1); an=symbols.get(counterion,counterion)
    try: ox=int(oxidation)
    except: ox=2
    import math
    g=math.gcd(abs(ox),abs(q)); nm=abs(q)//g; na=abs(ox)//g
    mf=metal+(str(nm) if nm>1 else '')
    af=f'({an}){na}' if na>1 and len(an)>1 else an+(str(na) if na>1 else '')
    h=float(hydration or 0)
    return mf+af+(f'·{h:g}H2O' if h>0 else '')
