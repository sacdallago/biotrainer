from __future__ import annotations

from typing import List, Tuple

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

def render_evaluate_view():
    st.subheader("Evaluate")
    
    st.write("Learn how to run autoeval.")

    st.markdown("- [Autoeval Docs](https://github.com/sacdallago/biotrainer/blob/main/docs/autoeval.md) - Autoeval Documentation.")
    st.markdown("- [Autoeval Example Notebooks](https://github.com/sacdallago/biotrainer/tree/main/examples/autoeval) - Get started with autoeval example notebooks.")
