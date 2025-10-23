from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


_WHATS_NEW: list[tuple[str, str]] = [
    ("🚀", "Groq-powered conversations now guide chart building and data wrangling."),
    ("🧠", "Integrated YData Profiling reports provide instant, shareable data audits."),
    ("🎨", "A refreshed layout and cohesive theming keeps key actions within reach."),
]

_FEATURE_CALLOUTS: list[tuple[str, str]] = [
    ("🔍", "Sketch + Groq insights for natural language exploration."),
    ("📊", "Plotly templates, metrics, and filtering ready out of the box."),
    ("🤝", "Collaborative-ready downloads and shareable profiling reports."),
]


def _render_list(items: Iterable[tuple[str, str]], columns: int = 3) -> None:
    cols = st.columns(columns)
    for idx, (icon, text) in enumerate(items):
        with cols[idx % columns]:
            st.markdown(f"{icon} {text}")


def _summarise_dataframe(df: pd.DataFrame | None) -> tuple[str, str, str]:
    if df is None or df.empty:
        return "—", "—", "—"

    row_count = f"{df.shape[0]:,}"
    column_count = f"{df.shape[1]:,}"
    numeric_cols = df.select_dtypes(include="number").shape[1]
    return row_count, column_count, str(numeric_cols)


def render_home(personal_info: str, current_df: pd.DataFrame | None = None) -> None:
    """Render the redesigned home view for the Streamlit app."""
    summary = personal_info.split("\n\n")[0].strip() if personal_info else (
        "I build pragmatic machine learning and analytics experiences for teams that need fast, reliable insight."
    )

    hero = st.container()
    with hero:
        st.markdown("## Hi, I'm Samson Tan Jia Sheng 👋")
        st.markdown("### Data scientist crafting human-friendly analytics journeys.")
        st.write(summary)

        cta_columns = st.columns(2)
        with cta_columns[0]:
            st.markdown("[📁 Explore my portfolio](https://github.com/samsontands)")
        with cta_columns[1]:
            st.markdown("[💼 Connect on LinkedIn](https://www.linkedin.com/in/samsonthedatascientist/)")

    st.divider()

    st.subheader("What's new")
    st.caption("Highlights from the latest release.")
    _render_list(_WHATS_NEW, columns=1)

    st.divider()

    st.subheader("Your data at a glance")
    rows, cols, numeric = _summarise_dataframe(current_df)
    metrics = st.columns(3)
    metrics[0].metric("Rows loaded", rows)
    metrics[1].metric("Columns available", cols)
    metrics[2].metric("Numeric fields", numeric)
    if rows == "—":
        st.info("Upload a dataset from the sidebar to unlock instant metrics and chart templates.")

    st.divider()

    st.subheader("Toolkit highlights")
    _render_list(_FEATURE_CALLOUTS, columns=3)

    st.caption(
        "Ready to dive deeper? Use the navigation to explore the dataframe, launch guided visualisations, or chat with the AI assistant."
    )
