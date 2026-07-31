from urllib.parse import quote
import requests

def resolve_name(query,timeout=8):
    q=quote(str(query).strip(),safe='')
    url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/Title,MolecularFormula,CanonicalSMILES,IsomericSMILES,InChIKey,MolecularWeight/JSON'
    r=requests.get(url,timeout=timeout,headers={'User-Agent':'MOF-Synthesis-Assistant/8.0'})
    r.raise_for_status()
    return r.json()['PropertyTable']['Properties'][0]
