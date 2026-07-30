from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

LABELS = {
    0: "Fallimento / nessun MOF",
    1: "Prodotto amorfo o scarsamente cristallino",
    2: "MOF cristallino",
}

CLASS_SHORT = {0: "Fallimento", 1: "Amorfo", 2: "Cristallino"}
