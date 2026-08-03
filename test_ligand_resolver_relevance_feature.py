import sys
sys.path.insert(0, ".")
from unittest.mock import patch, MagicMock

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

from src.literature import _result_is_relevant, _significant_tokens, discover_ligand_identifiers

print("=== 1. _significant_tokens: esclude parole generiche e numeri nudi ===")
tokens = _significant_tokens("3,3'-amino-4,4'-bipyrazole ligand MOF linker synonym CAS SMILES chemical name")
check("1.1 le parole di rumore iniettate nella query di ricerca sono escluse",
      not ({"ligand","mof","linker","synonym","cas","smiles","chemical","name"} & tokens), f"{tokens}")
check("1.2 i locanti numerici nudi (3, 4) sono esclusi", not ({"3","4"} & tokens), f"{tokens}")
check("1.3 le parole distintive restano", {"amino","bipyrazole"} <= tokens, f"{tokens}")

print()
print("=== 2. _result_is_relevant: il caso concreto 'Toxin C2' segnalato dall'utente ===")
query_tokens = _significant_tokens("3,3'-amino-4,4'-bipyrazole")
toxin_text = ("Toxin C2 (C10H17N7O11S2) is a sulfamic acid derivative studied for its "
              "biological activity. The amino groups play a role in binding.")
check("2.1 il testo che ha prodotto 'Toxin C2' viene ora respinto",
      _result_is_relevant(toxin_text, query_tokens) is False)

relevant_text = "Synthesis of 3,3'-diamino-4,4'-bipyrazole as a linker for coordination polymers."
check("2.2 un testo genuinamente pertinente viene accettato",
      _result_is_relevant(relevant_text, query_tokens) is True)

check("2.3 query senza token distintivi (es. solo una formula) non blocca nulla (fallback permissivo)",
      _result_is_relevant("qualunque testo", _significant_tokens("C6H7N5")) is True)

print()
print("=== 3. Test end-to-end con risposta Tavily simulata (mock), replicando lo scenario dello screenshot ===")

def _mock_tavily_response(results):
    client = MagicMock()
    client.search.return_value = {"results": results}
    return client

with patch("tavily.TavilyClient") as MockClient:
    MockClient.return_value = _mock_tavily_response([
        {"title": "Toxin C2 sulfamic acid derivative", "content": "Toxin C2 is a sulfamic acid derivative with amino groups involved in binding."},
    ])
    with patch("src.literature._api_key", return_value="fake-key-for-test"):
        out = discover_ligand_identifiers("3,3'-amino-4,4'-bipyrazole")
    check("3.1 il risultato irrilevante non produce più alcun identificatore spurio", out == [], f"{out}")

with patch("tavily.TavilyClient") as MockClient:
    MockClient.return_value = _mock_tavily_response([
        {"title": "3,3'-diamino-4,4'-bipyrazole (H2bpz-NH2)",
         "content": "Synthesis of the ligand 3,3'-diamino-4,4'-bipyrazole, abbreviated (H2bpzNH2), CAS 999-88-7, used as a linker."},
    ])
    with patch("src.literature._api_key", return_value="fake-key-for-test"):
        out = discover_ligand_identifiers("3,3'-amino-4,4'-bipyrazole")
    check("3.2 un risultato pertinente continua a produrre identificatori utili", len(out) > 0, f"{out}")

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
