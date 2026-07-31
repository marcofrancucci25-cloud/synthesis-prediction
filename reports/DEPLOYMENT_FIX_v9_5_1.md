# Deployment fix v9.5.1

- RDKit Draw is now imported lazily, so missing optional Linux graphics libraries cannot crash the app at startup.
- Added Streamlit Cloud system packages for RDKit 2D rendering: libxrender1, libxext6, libsm6 and libgl1.
- When rendering is unavailable, the identity card and molecular identifiers remain fully functional.
