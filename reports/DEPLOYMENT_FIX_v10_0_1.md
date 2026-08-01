# Deployment fix v10.0.1

The application now imports `src.engine` as a module and resolves the joint optimizer at runtime.
This avoids a fatal startup ImportError if Streamlit temporarily deploys `app.py` before the matching
`src/engine.py` revision. The complete `src` directory must still be replaced as one release.
