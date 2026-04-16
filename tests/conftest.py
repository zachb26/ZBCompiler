# conftest.py — adds repo root to sys.path and stubs heavy optional deps
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub streamlit so modules that do `import streamlit as st` at the top level
# can be imported in a headless test environment without streamlit installed.
if "streamlit" not in sys.modules:
    st_stub = types.ModuleType("streamlit")
    st_stub.session_state = {}
    st_stub.cache_data = lambda *a, **kw: (lambda f: f)  # no-op decorator
    st_stub.cache_resource = lambda *a, **kw: (lambda f: f)
    sys.modules["streamlit"] = st_stub
