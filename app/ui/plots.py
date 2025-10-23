"""Visualization helpers for the grapher tab."""
from __future__ import annotations

import traceback
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit_antd_components as sac
from streamlit_extras.grid import grid
from streamlit_extras.no_default_selectbox import selectbox
from wordcloud import WordCloud


COLORSCALES = {
    "Plotly3": px.colors.sequential.Plotly3,
    "Viridis": px.colors.sequential.Viridis,
    "Cividis": px.colors.sequential.Cividis,
    "Inferno": px.colors.sequential.Inferno,
    "Magma": px.colors.sequential.Magma,
    "Plasma": px.colors.sequential.Plasma,
    "Turbo": px.colors.sequential.Turbo,
    "Blackbody": px.colors.sequential.Blackbody,
    "Bluered": px.colors.sequential.Bluered,
    "Electric": px.colors.sequential.Electric,
    "Jet": px.colors.sequential.Jet,
    "Rainbow": px.colors.sequential.Rainbow,
    "Blues": px.colors.sequential.Blues,
    "BuGn": px.colors.sequential.BuGn,
    "BuPu": px.colors.sequential.BuPu,
    "GnBu": px.colors.sequential.GnBu,
    "Greens": px.colors.sequential.Greens,
    "Greys": px.colors.sequential.Greys,
    "OrRd": px.colors.sequential.OrRd,
    "Oranges": px.colors.sequential.Oranges,
    "PuBu": px.colors.sequential.PuBu,
    "PuBuGn": px.colors.sequential.PuBuGn,
    "PuRd": px.colors.sequential.PuRd,
    "Purples": px.colors.sequential.Purples,
    "RdBu": px.colors.sequential.RdBu,
    "RdPu": px.colors.sequential.RdPu,
    "Reds": px.colors.sequential.Reds,
    "YlOrBr": px.colors.sequential.YlOrBr,
    "YlOrRd": px.colors.sequential.YlOrRd,
    "turbid": px.colors.sequential.turbid,
    "thermal": px.colors.sequential.thermal,
    "haline": px.colors.sequential.haline,
    "solar": px.colors.sequential.solar,
    "ice": px.colors.sequential.ice,
    "gray": px.colors.sequential.gray,
    "deep": px.colors.sequential.deep,
    "dense": px.colors.sequential.dense,
    "algae": px.colors.sequential.algae,
    "matter": px.colors.sequential.matter,
    "speed": px.colors.sequential.speed,
    "amp": px.colors.sequential.amp,
    "tempo": px.colors.sequential.tempo,
    "Burg": px.colors.sequential.Burg,
    "Burgyl": px.colors.sequential.Burgyl,
    "Redor": px.colors.sequential.Redor,
    "Oryel": px.colors.sequential.Oryel,
    "Peach": px.colors.sequential.Peach,
    "Pinkyl": px.colors.sequential.Pinkyl,
    "Mint": px.colors.sequential.Mint,
    "Blugrn": px.colors.sequential.Blugrn,
    "Darkmint": px.colors.sequential.Darkmint,
    "Emrld": px.colors.sequential.Emrld,
    "Aggrnyl": px.colors.sequential.Aggrnyl,
    "Bluyl": px.colors.sequential.Bluyl,
    "Teal": px.colors.sequential.Teal,
    "Tealgrn": px.colors.sequential.Tealgrn,
    "Purp": px.colors.sequential.Purp,
    "Purpor": px.colors.sequential.Purpor,
    "Sunset": px.colors.sequential.Sunset,
    "Magenta": px.colors.sequential.Magenta,
    "Sunsetdark": px.colors.sequential.Sunsetdark,
    "Agsunset": px.colors.sequential.Agsunset,
    "Brwnyl": px.colors.sequential.Brwnyl,
}


def render_grapher_page(st_module: Any, curr_filtered_df: pd.DataFrame) -> str:
    """Render the grapher segmented control and return any log output."""
    log = ""

    grapher_tabs = sac.segmented(
        items=[
            sac.SegmentedItem(label="Scatter"),
            sac.SegmentedItem(label="Line"),
            sac.SegmentedItem(label="Bar"),
            sac.SegmentedItem(label="Histogram"),
            sac.SegmentedItem(label="Box"),
            sac.SegmentedItem(label="Violin"),
            sac.SegmentedItem(label="Scatter 3D"),
            sac.SegmentedItem(label="Heatmap"),
            sac.SegmentedItem(label="Contour"),
            sac.SegmentedItem(label="Pie"),
            sac.SegmentedItem(label="Splom"),
            sac.SegmentedItem(label="Candlestick"),
            sac.SegmentedItem(label="Word Cloud"),
        ],
        label=None,
        position="top",
        index=0,
        format_func="title",
        radius="md",
        size="md",
        align="center",
        direction="horizontal",
        grow=True,
        disabled=False,
        readonly=False,
        return_index=True,
    )

    if not st_module.session_state.get("select_df"):
        return log

    if grapher_tabs == 0:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = st_module.multiselect(
                "**Select y values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_1",
                default=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_2",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_5",
                no_selection_label=None,
            )
            symbol = selectbox(
                "**Select symbol value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_6",
                no_selection_label=None,
            )
            size = selectbox(
                "**Select size value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_1_7",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_1_8",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if y:
                    fig = px.scatter(
                        data_frame=curr_filtered_df,
                        x=x,
                        y=y,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        symbol=symbol,
                        size=size,
                        render_mode="auto",
                        height=750,
                        color_discrete_sequence=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.scatter(height=750, render_mode="auto"),
                        use_container_width=True,
                    )
                log = ""
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.scatter(height=750, render_mode="auto"),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 1:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = st_module.multiselect(
                "**Select y values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_2_1",
                default=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_2_2",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_2_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_2_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_2_5",
                no_selection_label=None,
            )
            aggregation = selectbox(
                "**Select aggregation**",
                ["mean", "median", "min", "max", "sum"],
                key="grid_grapher_2_6",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_2_7",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                line_plot_df = curr_filtered_df.copy()
                key_cols_line = [
                    val
                    for val in [x, color, facet_row, facet_col]
                    if val is not None
                ]
                if key_cols_line:
                    if aggregation is not None:
                        line_plot_df = (
                            curr_filtered_df.groupby(key_cols_line)
                            .agg(aggregation)
                            .reset_index()
                        )
                    else:
                        line_plot_df = curr_filtered_df.sort_values(key_cols_line)
                if y:
                    fig = px.line(
                        data_frame=line_plot_df,
                        x=x,
                        y=y,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        render_mode="auto",
                        height=750,
                        color_discrete_sequence=COLORSCALES[plot_color],
                    )
                    fig.update_traces(connectgaps=True)
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.line(height=750, render_mode="auto"),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.line(height=750, render_mode="auto"),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 2:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = st_module.multiselect(
                "**Select y values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_3_1",
                default=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_3_2",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_3_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_3_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_3_5",
                no_selection_label=None,
            )
            aggregation = selectbox(
                "**Select aggregation**",
                ["mean", "median", "min", "max", "sum"],
                key="grid_grapher_3_6",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_3_7",
            )
            sort = selectbox(
                "**Select sort type**",
                ["asc", "desc"],
                key="grid_grapher_3_8",
                no_selection_label=None,
            )
        with grid_grapher.expander("", expanded=True):
            try:
                bar_plot_df = curr_filtered_df.copy()
                key_cols_bar = [
                    val
                    for val in [x, color, facet_row, facet_col]
                    if val is not None
                ]
                if key_cols_bar:
                    if aggregation is not None:
                        bar_plot_df = (
                            curr_filtered_df.groupby(key_cols_bar)
                            .agg(aggregation)
                            .reset_index()
                        )
                    else:
                        bar_plot_df = curr_filtered_df.sort_values(key_cols_bar)
                if sort is not None and y:
                    bar_plot_df = bar_plot_df.sort_values(
                        y,
                        ascending=(sort == "asc"),
                    )
                if y:
                    fig = px.bar(
                        data_frame=bar_plot_df,
                        x=x,
                        y=y,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        height=750,
                        color_continuous_scale=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.bar(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.bar(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 3:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            x = st_module.multiselect(
                "**Select x values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_4_1",
                default=None,
            )
            color = selectbox(
                "**Select color values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_4_2",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_4_3",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_4_4",
                no_selection_label=None,
            )
            marginal = selectbox(
                "**Select marginal**",
                ["rug", "box", "violin"],
                key="grid_grapher_4_5",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_4_6",
            )
            cumulative = st_module.checkbox(
                "Cumulative ?",
                key="grid_grapher_4_7",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if x:
                    fig = px.histogram(
                        data_frame=curr_filtered_df,
                        x=x,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        marginal=marginal,
                        cumulative=cumulative,
                        height=750,
                        color_discrete_sequence=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.bar(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.bar(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 4:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = st_module.multiselect(
                "**Select y values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_5_1",
                default=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_5_2",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_5_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_5_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_5_5",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_5_6",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if y:
                    fig = px.box(
                        data_frame=curr_filtered_df,
                        x=x,
                        y=y,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        height=750,
                        color_discrete_sequence=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.box(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.box(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 5:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = st_module.multiselect(
                "**Select y values**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_6_1",
                default=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_6_2",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_6_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_6_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_6_5",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_6_6",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if y:
                    fig = px.violin(
                        data_frame=curr_filtered_df,
                        x=x,
                        y=y,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        height=750,
                        color_discrete_sequence=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.violin(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.violin(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 6:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            z = selectbox(
                "**Select z value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_7_1",
                no_selection_label=None,
            )
            y = selectbox(
                "**Select y value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_7_2",
                no_selection_label=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_7_3",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_7_4",
                no_selection_label=None,
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if x and y and z:
                    fig = px.scatter_3d(
                        data_frame=curr_filtered_df,
                        x=x,
                        y=y,
                        z=z,
                        color=color,
                        height=750,
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.scatter_3d(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.scatter_3d(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 7:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = selectbox(
                "**Select y value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_8_1",
                no_selection_label=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_8_2",
                no_selection_label=None,
            )
            z = selectbox(
                "**Select z value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_8_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_8_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_8_5",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_8_6",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if x and y:
                    fig = px.density_heatmap(
                        data_frame=curr_filtered_df,
                        x=x,
                        y=y,
                        z=z,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        height=750,
                        color_continuous_scale=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.density_heatmap(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.density_heatmap(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 8:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            y = selectbox(
                "**Select y value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_9_1",
                no_selection_label=None,
            )
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_9_2",
                no_selection_label=None,
            )
            z = selectbox(
                "**Select z value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_9_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_9_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_9_5",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_9_6",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if y:
                    fig = px.density_contour(
                        data_frame=curr_filtered_df,
                        x=x,
                        y=y,
                        color=z,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        height=750,
                    )
                    fig.update_traces(
                        contours_coloring="fill",
                        contours_showlabels=True,
                        colorscale=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.density_contour(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.density_contour(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 9:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            name = selectbox(
                "**Select name value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_10_1",
                no_selection_label=None,
            )
            value = selectbox(
                "**Select value's value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_10_2",
                no_selection_label=None,
            )
            color = selectbox(
                "**Select color value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_10_3",
                no_selection_label=None,
            )
            facet_row = selectbox(
                "**Select facet row value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_10_4",
                no_selection_label=None,
            )
            facet_col = selectbox(
                "**Select facet col value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_10_5",
                no_selection_label=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_10_6",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if name:
                    fig = px.pie(
                        data_frame=curr_filtered_df,
                        names=name,
                        values=value,
                        color=color,
                        facet_row=facet_row,
                        facet_col=facet_col,
                        height=750,
                        color_discrete_sequence=COLORSCALES[plot_color],
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.pie(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.pie(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 10:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            dimensions = st_module.multiselect(
                "**Select dimensions value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_11_1",
                default=None,
            )
            color = selectbox(
                "**Select color value (Column should be included as one of the dimension value)**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_11_2",
                no_selection_label=None,
            )
            diag = st_module.selectbox(
                "**Select Diagonal Plot**",
                ["scatter", "histogram", "box"],
                index=1,
                key="grid_grapher_11_3",
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                [
                    "Greys",
                    "YlGnBu",
                    "Greens",
                    "YlOrRd",
                    "Bluered",
                    "RdBu",
                    "Reds",
                    "Blues",
                    "Picnic",
                    "Rainbow",
                    "Portland",
                    "Jet",
                    "Hot",
                    "Blackbody",
                    "Earth",
                    "Electric",
                    "Viridis",
                    "Cividis",
                ],
                index=0,
                key="grid_grapher_11_4",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if dimensions:
                    fig = ff.create_scatterplotmatrix(
                        curr_filtered_df[dimensions],
                        diag=diag,
                        title="",
                        index=color,
                        colormap=plot_color,
                        height=750,
                    )
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.bar(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.bar(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 11:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            x = selectbox(
                "**Select x value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_12_1",
                no_selection_label=None,
            )
            open_ = selectbox(
                "**Select open value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_12_2",
                no_selection_label=None,
            )
            high = selectbox(
                "**Select high value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_12_3",
                no_selection_label=None,
            )
            low = selectbox(
                "**Select low value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_12_4",
                no_selection_label=None,
            )
            close = selectbox(
                "**Select close value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_12_5",
                no_selection_label=None,
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if x and open_ and high and low and close:
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=curr_filtered_df[x],
                                open=curr_filtered_df[open_],
                                high=curr_filtered_df[high],
                                low=curr_filtered_df[low],
                                close=curr_filtered_df[close],
                            )
                        ]
                    )
                    fig.update_layout(height=750)
                    st_module.plotly_chart(fig, use_container_width=True)
                else:
                    st_module.plotly_chart(
                        px.density_contour(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.density_contour(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    elif grapher_tabs == 12:
        grid_grapher = grid([1, 2], vertical_align="bottom")
        with grid_grapher.expander(label="Features", expanded=True):
            words = st_module.multiselect(
                "**Select words value**",
                curr_filtered_df.columns.to_list(),
                key="grid_grapher_13_1",
                default=None,
            )
            plot_color = st_module.selectbox(
                "**Select Plot Color Map**",
                list(COLORSCALES.keys()),
                index=0,
                key="grid_grapher_13_2",
            )
        with grid_grapher.expander("", expanded=True):
            try:
                if words:
                    if isinstance(words, str):
                        words = [words]
                    text = " ".join(
                        pd.concat(
                            [
                                curr_filtered_df[column]
                                .dropna()
                                .astype(str)
                                for column in words
                            ]
                        )
                    )
                    wc = WordCloud(scale=2, collocations=False).generate(text)
                    st_module.plotly_chart(
                        px.imshow(wc, color_continuous_scale=COLORSCALES[plot_color]),
                        height=750,
                        use_container_width=True,
                    )
                else:
                    st_module.plotly_chart(
                        px.bar(height=750),
                        use_container_width=True,
                    )
            except Exception:  # pragma: no cover - visualization issues
                st_module.plotly_chart(
                    px.bar(height=750),
                    use_container_width=True,
                )
                log = traceback.format_exc()
        st_module.subheader("**Console Log**", anchor=False)
        st_module.markdown(f"{log}")

    return log


__all__ = ["render_grapher_page"]
