from src.assets import load_assets
from src.engine import MOFSynthesisEngine

def main():
    assets = load_assets()
    engine = MOFSynthesisEngine(assets)
    db = assets.database
    row = {f: db.iloc[0][f] for f in assets.schema["feature_order"]}
    result = engine.predict(row)
    assert len(result.probabilities) == 3
    assert abs(float(result.probabilities.sum()) - 1.0) < 1e-6
    optimized = engine.optimize(row, 5, True, True)
    assert len(optimized) == 5
    print("Smoke test superato.")

if __name__ == "__main__":
    main()
