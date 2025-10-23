"""User-interface helpers for the profiling tab."""
from __future__ import annotations

from typing import Any

from ydata_profiling import ProfileReport


def show_eda_tool(st_module: Any) -> None:
    """Render the YData profiling workflow."""
    st_module.title("Data Profiling with YData Profiling")

    session = st_module.session_state
    if session.get("select_df"):
        df = session["filtered_df"]
        st_module.write(df)

        if st_module.button("Generate Profiling Report"):
            with st_module.spinner("Generating profiling report..."):
                profile = ProfileReport(
                    df,
                    title="Pandas Profiling Report",
                    explorative=True,
                )
                report_html = profile.to_html()

            st_module.success("Report generated successfully!")
            st_module.download_button(
                label="Download Profiling Report",
                data=report_html,
                file_name="profiling_report.html",
                mime="text/html",
            )
    else:
        st_module.warning("Please select a dataframe from the sidebar first.")


__all__ = ["show_eda_tool"]
