# Deployment fix v10.7.1

## Observed failure

Streamlit raised `AttributeError` at application startup on:

```python
verified_precedents = engine.verified_precedents
```

The traceback proves that the deployed `app.py` was newer than the imported `src/engine.py`, or that Streamlit was still serving a cached engine module from the previous revision.

## Correction

The evidence function is now resolved lazily with `getattr`. During a temporary mixed-revision state, only the optional evidence panel is unavailable; prediction, validity checks, optimization and literature search continue to load normally. Once the complete v10.7.1 files are synchronized, verified precedents are restored automatically.

The predictive model and its probabilities were not changed by this hotfix.
