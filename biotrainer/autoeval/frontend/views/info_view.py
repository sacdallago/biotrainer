from __future__ import annotations

from typing import List, Tuple

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise


def render_info_view():
    st.subheader("About Autoeval")
    st.write(
        "The `autoeval` module facilitates the automatic evaluation of protein language models "
        "on downstream tasks. It is designed to allow quick insights into downstream performance of protein"
        "language models after training."
    )
    st.markdown("### Supported Datasets")
    st.markdown("#### PBC (Supervised)")
    st.write(
        "- [ProteinBenchmarkCollection](https://github.com/Rostlab/pbc): Supervised tasks such as:\n"
        "  - **scl**: Protein subcellular localization prediction.\n"
        "  - **secondary_structure**: Protein secondary structure prediction, including various test sets."
    )
    st.markdown("#### PGYM (Zero-Shot)")
    st.write(
        "- [ProteinGym DMS Supervised](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip) - Datasets with deep mutational scanning fitness scores for protein mutations, split into:\n"
        "  - **virus**: Viral protein datasets.\n"
        "  - **non-virus**: Non-viral protein datasets.\n"
        "  - **total**: Combination of viral and non-viral datasets."
    )
