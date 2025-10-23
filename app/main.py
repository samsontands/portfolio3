from __future__ import annotations

import json
import os
import re
import sqlite3
import traceback
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac
import pygwalker as pyg
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_lottie import st_lottie

from app.data.loader import (
    DataLoaderError,
    ensure_state_defaults,
    fetch_github_csv,
    prepare_default_data,
    prepare_uploaded_data,
    update_state_with_data,
)
from app.services.ai import (
    AIServiceError,
    extract_code,
    generate_gpt_response,
    get_groq_response,
)
from app.ui.plots import render_grapher_page
from app.ui.profiling import show_eda_tool
from config.settings import (
    CSV_CACHE_TTL,
    DEFAULT_CSV_NAME,
    DEFAULT_CSV_URL,
    LOTTIE_DIR,
    PERSONAL_INFO_PATH,
    SYSTEM_PROMPT_PATH,
)


class MenuPage(IntEnum):
    """Enumeration mirroring the order of sidebar menu items."""

    HOME = 0
    DATAFRAME = 1
    STATISTICS = 2
    GRAPHER = 3
    PYGWALKER = 4
    ASK_AI = 5
    PROJECTS = 6
    ASK_ME_ANYTHING = 7
    PROFILING = 8


@dataclass(frozen=True)
class Project:
    """Metadata for a project card displayed on the portfolio page."""

    title: str
    description: str
    image: str
    url: str


PROJECTS: tuple[Project, ...] = tuple(
    sorted(
        (
            Project(
                title="Agentic Retrieval Augmented Generation (RAG)",
                description="A webpage for AI with tool use and fact checking",
                image="https://images.prismic.io/codiste-website/08ac7396-b806-4550-b167-8814f6eb0fe2_What+is+the+difference+between+GPT_+GPT3%2C+GPT+3.5%2C+GPT+turbo+and+GPT-4.png?auto=compress,format",
                url="https://agent-rag-ai.streamlit.app/",
            ),
            Project(
                title="Depcreciation Analysis Demo",
                description="A website to showcase an analysis to calculate vehicle depreciation",
                image="https://static.vecteezy.com/system/resources/previews/005/735/523/original/thin-line-car-icons-set-in-black-background-universal-car-icon-to-use-in-web-and-mobile-ui-car-basic-ui-elements-set-free-vector.jpg",
                url="https://depreciationanalysis.streamlit.app/",
            ),
            Project(
                title="File Transfer App",
                description="A webpage for temporary file transfer.",
                image="https://img.freepik.com/premium-photo/cloud-storage-icon-neon-element-black-background-3d-rendering-illustration_567294-1378.jpg?w=740",
                url="https://filecpdi.streamlit.app/",
            ),
            Project(
                title="Website Scraper POC",
                description="A website to showcase web scraper proof of concept",
                image="https://miro.medium.com/v2/resize:fit:720/format:webp/1*nKwYuOo-zhF8eHocsR9WvA.png",
                url="https://scraperpoc.streamlit.app/",
            ),
        ),
        key=lambda project: project.title,
    )
)

DATA_PAGES: set[MenuPage] = {
    MenuPage.DATAFRAME,
    MenuPage.STATISTICS,
    MenuPage.GRAPHER,
    MenuPage.PYGWALKER,
    MenuPage.ASK_AI,
    MenuPage.PROFILING,
}

TYPE_OPTIONS: tuple[str, ...] = ("int64", "float64", "str", "bool", "object", "timestamp")
SIDEBAR_ANIMATION = LOTTIE_DIR / "Animation - 1694990107205.json"
HOME_ANIMATIONS: dict[str, Path] = {
    "hero": LOTTIE_DIR / "Animation - 1694988603751.json",
    "projects_left": LOTTIE_DIR / "Animation - 1694988937837.json",
    "projects_right": LOTTIE_DIR / "Animation - 1694989926620.json",
    "contact": LOTTIE_DIR / "Animation - 1694990540946.json",
    "future": LOTTIE_DIR / "Animation - 1694991370591.json",
}


def configure_environment() -> None:
    """Prepare environment variables so Sketch never calls OpenAI directly."""

    os.environ.setdefault("SKETCH_MAX_COLUMNS", "50")
    os.environ["SKETCH_USE_REMOTE_LAMBDAPROMPT"] = "True"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("LAMBDAPROMPT_BACKEND", None)
    os.environ.pop("LAMBDAPROMPT_OPENAI_MODEL", None)


@st.cache_data(show_spinner=False, ttl=CSV_CACHE_TTL)
def load_default_csv(url: str, token: str | None = None) -> pd.DataFrame:
    """Fetch the default CSV file bundled with the app (or warn on failure)."""

    try:
        return fetch_github_csv(url, token=token)
    except DataLoaderError as exc:
        st.warning(str(exc))
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_lottiefile(path_str: str) -> dict[str, Any] | None:
    """Load and cache a Lottie animation, returning ``None`` if unavailable."""

    path = Path(path_str)
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        st.warning(f"Animation file not found: {path.name}")
    except json.JSONDecodeError as exc:
        st.warning(f"Invalid Lottie animation {path.name}: {exc}")
    except OSError as exc:
        st.warning(f"Unable to read animation {path.name}: {exc}")
    return None


@st.cache_data(show_spinner=False)
def load_personal_info() -> str:
    """Read the portfolio biography from disk."""

    try:
        return PERSONAL_INFO_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning("Personal information file not found.")
    except OSError as exc:
        st.warning(f"Unable to read personal information: {exc}")
    return ""


@st.cache_data(show_spinner=False)
def load_system_prompt() -> str:
    """Read the Groq system prompt from disk."""

    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning("System prompt file not found.")
    except OSError as exc:
        st.warning(f"Unable to read system prompt: {exc}")
    return ""


def sidebar_animation() -> None:
    """Render the sidebar animation (cached JSON keeps it snappy)."""

    animation = load_lottiefile(str(SIDEBAR_ANIMATION))
    if animation:
        st_lottie(animation, key="sidebar_animation")


def convert_df(df: pd.DataFrame, *, index: bool = False) -> bytes:
    """Serialize a DataFrame to CSV bytes for download buttons."""

    return df.to_csv(index=index).encode("utf-8")


def render_sidebar() -> MenuPage:
    """Render sidebar controls and return the selected navigation page."""

    state = st.session_state
    with st.sidebar:
        sidebar_animation()

        with st.expander("Upload files", expanded=not state.get("file_uploaded", False)):
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["csv"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            loaded = prepare_uploaded_data(uploaded_files)
            if not loaded.has_data:
                gh_token = st.secrets.get("GITHUB_TOKEN", None)
                df_default = load_default_csv(DEFAULT_CSV_URL, token=gh_token)
                default_loaded = prepare_default_data(df_default, DEFAULT_CSV_NAME)
                if default_loaded.has_data:
                    loaded = default_loaded

            update_state_with_data(state, loaded)
            if loaded.has_data:
                selected = selectbox(
                    "**Select Dataframe**",
                    list(state["file_name"].keys()),
                    no_selection_label=None,
                )
                state["select_df"] = selected
                state["file_uploaded"] = True
            else:
                state["select_df"] = None
                state["filtered_df"] = pd.DataFrame()
                state["curr_filtered_df"] = pd.DataFrame()
                state["file_uploaded"] = False

        menu_index = sac.menu(
            [
                sac.MenuItem("Home", icon="house"),
                sac.MenuItem("DataFrame", icon="speedometer2"),
                sac.MenuItem("Statistics", icon="plus-slash-minus"),
                sac.MenuItem("Grapher", icon="graph-up"),
                sac.MenuItem("PygWalker", icon="plugin"),
                sac.MenuItem("Ask AI", icon="robot"),
                sac.MenuItem("My Projects", icon="card-text"),
                sac.MenuItem("Ask Me Anything", icon="chat-dots"),
                sac.MenuItem("YData Profiling", icon="bar-chart-line"),
            ],
            index=int(state.get("active_page", MenuPage.HOME)),
            format_func="title",
            size="small",
            indent=15,
            open_index=None,
            open_all=True,
            return_index=True,
        )
        state["active_page"] = menu_index

        st.markdown("### Contact Information")
        st.markdown("📞 +6011-1122 1128")
        st.markdown("✉️ samsontands@gmail.com")
        st.markdown("📍 Kuala Lumpur, Malaysia")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/samsonthedatascientist/)")

    return MenuPage(menu_index)


def _infer_type_option(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "timestamp"
    if is_bool_dtype(series):
        return "bool"
    if is_integer_dtype(series):
        return "int64"
    if is_float_dtype(series):
        return "float64"
    if series.dtype == object:
        return "object"
    return "str"


def _convert_column(series: pd.Series, target_type: str) -> pd.Series:
    if target_type == "timestamp":
        return pd.to_datetime(series, errors="coerce")
    if target_type == "str":
        return series.astype("string")
    if target_type == "int64":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.astype("Int64")
    if target_type == "float64":
        return pd.to_numeric(series, errors="coerce")
    if target_type == "bool":
        return series.astype("bool")
    if target_type == "object":
        return series.astype("object")
    return series


def render_data_filters() -> tuple[pd.DataFrame, str]:
    """Render the filter controls and return the filtered DataFrame and logs."""

    state = st.session_state
    log_messages: list[str] = []
    current = pd.DataFrame()

    if not state.get("select_df"):
        state["filtered_df"] = pd.DataFrame()
        state["curr_filtered_df"] = pd.DataFrame()
        return current, ""

    with st.expander("Filters"):
        try:
            frame = state["files"][state["file_name"][state["select_df"]]].copy()
        except Exception:
            log = traceback.format_exc()
            state["filtered_df"] = pd.DataFrame()
            state["curr_filtered_df"] = pd.DataFrame()
            return pd.DataFrame(), log

        working = frame.drop(columns=["Row_Number_"], errors="ignore")
        type_defaults = [_infer_type_option(working[column]) for column in working.columns]

        columns_config = st.data_editor(
            pd.DataFrame(
                {
                    "Column Name": working.columns.to_list(),
                    "Show?": True,
                    "Convert Type": type_defaults,
                }
            ),
            column_config={
                "Convert Type": st.column_config.SelectboxColumn(
                    "Convert Type",
                    options=TYPE_OPTIONS,
                    required=True,
                    default="object",
                )
            },
            num_rows="fixed",
            hide_index=True,
            height=250,
            use_container_width=True,
        )

        for column_name, target_type in zip(
            columns_config["Column Name"], columns_config["Convert Type"]
        ):
            try:
                working[column_name] = _convert_column(working[column_name], target_type)
            except Exception as exc:  # pragma: no cover - UI feedback path
                log_messages.append(
                    f"Failed to convert column '{column_name}' to {target_type}: {exc}"
                )

        st.caption("**:red[Note:] Date / Time columns are converted to Timestamp when selected.**")

        try:
            filtered = dataframe_explorer(working, case=False)
        except Exception:  # pragma: no cover - UI feedback path
            log = traceback.format_exc()
            state["filtered_df"] = pd.DataFrame()
            state["curr_filtered_df"] = pd.DataFrame()
            return pd.DataFrame(), log

        filtered = filtered.drop(columns=["Row_Number_"], errors="ignore")
        visible_columns = columns_config.loc[columns_config["Show?"], "Column Name"].tolist()
        current = filtered[visible_columns] if visible_columns else filtered

        state["filtered_df"] = filtered
        state["curr_filtered_df"] = current

    return current, "\n".join(log_messages)


def render_dataframe_page(curr_filtered_df: pd.DataFrame, log: str) -> None:
    state = st.session_state
    if not state.get("select_df"):
        st.info("Upload or select a dataset from the sidebar to explore it here.")
        return

    st.data_editor(
        curr_filtered_df,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=False,
    )
    st.caption("**:red[Note:] To delete rows, press the delete key after selecting rows.**")
    st.markdown(
        f"**DataFrame Shape: {curr_filtered_df.shape[0]} x {curr_filtered_df.shape[1]}**"
    )
    st.download_button(
        label="**Download Modified DataFrame as CSV**",
        data=convert_df(curr_filtered_df),
        file_name=f"{state['select_df']}",
        mime="text/csv",
    )
    st.subheader("**Console Log**", anchor=False)
    if log:
        st.markdown(log)
    else:
        st.caption("No conversion issues detected.")


def render_statistics_page(curr_filtered_df: pd.DataFrame, log: str) -> None:
    state = st.session_state
    if not state.get("select_df"):
        st.info("Upload or select a dataset from the sidebar to review statistics.")
        return

    if curr_filtered_df.empty:
        st.warning("Your filtered DataFrame is empty. Adjust filters or upload data.")
        return

    stats = curr_filtered_df.describe(include="all").transpose()
    stats["Unique"] = curr_filtered_df.apply(lambda column: column.nunique(dropna=True))
    st.dataframe(stats, use_container_width=True)
    st.markdown(f"**DataFrame Shape: {curr_filtered_df.shape[0]} x {curr_filtered_df.shape[1]}**")
    st.download_button(
        label="**Download Statistics DataFrame as CSV**",
        data=convert_df(stats, index=True),
        file_name=f"stats_{state['select_df']}",
        mime="text/csv",
    )
    st.subheader("**Console Log**", anchor=False)
    if log:
        st.markdown(log)
    else:
        st.caption("No conversion issues detected.")


def render_pygwalker_page(curr_filtered_df: pd.DataFrame) -> None:
    state = st.session_state
    if not state.get("select_df"):
        st.info("Upload or select a dataset from the sidebar to launch PygWalker.")
        return

    if curr_filtered_df.empty:
        st.warning("Your filtered DataFrame is empty. Adjust filters before continuing.")
        return

    st.markdown("**Are you sure you want to launch the PygWalker interface?**")
    if st.button("Continue", key="pygwalker"):
        try:
            pyg.walk(curr_filtered_df, env="Streamlit", dark="media")
        except Exception:  # pragma: no cover - external library rendering
            st.error("Unable to launch PygWalker with the current dataset.")
            st.code(traceback.format_exc())


def _render_sql_assistant(df_ai: pd.DataFrame) -> None:
    question = st.text_area(
        "Ask a concise question. Example: What is the average pay of the female?",
        value="What is the average pay of the female?",
    )
    if st.button("Run SQL"):
        try:
            with sqlite3.connect(":memory:") as conn:
                table_name = "my_table"
                df_ai.to_sql(table_name, conn, if_exists="replace", index=False)

                cols = ", ".join(df_ai.columns.tolist())
                prompt = (
                    "Write a SQLite query based on this question: {question} "
                    "The table name is my_table and the table has the following columns: {cols}. "
                    "Return only a SQL query and nothing else."
                ).format(question=question, cols=cols)

                sql = generate_gpt_response(
                    prompt,
                    max_tokens=250,
                    secrets=st.secrets,
                )
                sql = extract_code(sql)

                with st.expander("SQL used"):
                    st.code(sql, language="sql")

                out = pd.read_sql_query(sql, conn)
        except AIServiceError as exc:
            st.error(str(exc))
            return
        except Exception:
            st.error("Failed to run the generated SQL. Try rephrasing your question.")
            st.code(traceback.format_exc())
            return

        if out.shape == (1, 1):
            st.subheader(f"Answer: {out.iloc[0, 0]}")
        else:
            st.subheader("Result")
            st.dataframe(out, use_container_width=True)


def _render_chart_assistant(df_ai: pd.DataFrame) -> None:
    viz_req = st.text_area(
        "Describe the chart you want. Example: Create a pie chart of male vs female.",
        value="Create a pie chart of male vs female.",
    )

    if st.button("Generate chart"):
        try:
            cols = ", ".join(df_ai.columns.tolist())
            prompt = (
                "Write code in Python using Plotly to address this request: {req} "
                "Use the existing dataframe variable named df that has the following columns: {cols}. "
                "Do not include any import statements. "
                "Do not use animation_group. "
                "Use a transparent background. "
                "Return only executable code, no explanations."
            ).format(req=viz_req, cols=cols)

            code = generate_gpt_response(
                prompt,
                max_tokens=1500,
                secrets=st.secrets,
            )
            code = extract_code(code)

            missing: list[str] = []
            for match in re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", code):
                if match not in df_ai.columns:
                    missing.append(match)
            if missing:
                st.warning(f"Columns not found in data: {sorted(set(missing))}")

            code = re.sub(
                r"(\b\w+\b)\.show\(\)",
                r"st.plotly_chart(\1, use_container_width=True)",
                code,
            )
            code = re.sub(
                r"plotly\.io\.show\((\b\w+\b)\)",
                r"st.plotly_chart(\1, use_container_width=True)",
                code,
            )

            with st.expander("Code used"):
                st.code(code, language="python")

            local_scope = {"st": st, "px": px, "go": go, "np": np, "pd": pd, "df": df_ai}
            exec(code, {}, local_scope)
        except AIServiceError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Chart generation failed. Try a simpler description or different columns.")
            st.code(traceback.format_exc())


def render_ai_assistant_page(filtered_df: pd.DataFrame) -> None:
    st.title("Ask Your Data (AI) 🔎")
    st.caption(
        "Type a question and I’ll generate a SQLite query against the filtered DataFrame — "
        "or ask me to create a Plotly chart."
    )

    if not st.session_state.get("select_df"):
        st.warning("Please select a dataframe in the sidebar first.")
        return

    df_ai = filtered_df.copy().reset_index(drop=True)
    if df_ai.empty:
        st.warning("Your filtered DataFrame is empty. Adjust filters or upload data.")
        return

    df_ai.columns = df_ai.columns.str.replace(" ", "_", regex=False)
    sql_tab, chart_tab = st.tabs(["Ask (SQL)", "Create a chart"])
    with sql_tab:
        _render_sql_assistant(df_ai)
    with chart_tab:
        _render_chart_assistant(df_ai)


def render_projects_page() -> None:
    st.title("My Projects")
    st.markdown(
        """
        <style>
        .project-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            cursor: pointer;
        }
        .project-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .project-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .project-description {
            font-size: 0.9rem;
            color: #555;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    for index, project in enumerate(PROJECTS):
        target_column = col1 if index % 2 == 0 else col2
        with target_column:
            st.markdown(
                f"""
                <a href="{project.url}" target="_blank" style="text-decoration: none; color: inherit;">
                    <div class="project-card">
                        <img src="{project.image}" style="width:100%; border-radius:5px; margin-bottom:0.5rem;"/>
                        <div class="project-title">{project.title}</div>
                        <div class="project-description">{project.description}</div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)


def render_ask_me_anything_page() -> None:
    st.title("Ask Me Anything")
    st.write(
        "Ask a question to get a brief response about the creator's background, skills, or experience."
    )

    personal_info = load_personal_info()
    system_prompt = load_system_prompt()
    if not personal_info or not system_prompt:
        st.info("Add personal information and a system prompt to enable this feature.")
        return

    user_question = st.text_input("What would you like to know?")
    if user_question:
        with st.spinner("Getting a quick answer..."):
            try:
                response = get_groq_response(
                    user_question,
                    system_prompt,
                    personal_info,
                    secrets=st.secrets,
                )
                st.write(response)
            except AIServiceError as exc:
                st.error(str(exc))
    st.caption(
        "Note: Responses are kept brief. For more detailed information, please refer to other sections of the app."
    )


def render_home() -> None:
    st.divider()
    col_main, col_animation = st.columns([5, 1])
    with col_main:
        st.markdown(
            """
            ##### Hi, I am Samson Tan Jia Sheng 👋
            #### A Data Scientist From Malaysia
            **I am passionate about Data Analysis, Data Visualization, Machine Learning, and AI advancements.**
            """
        )
        personal_info = load_personal_info()
        if personal_info:
            first_paragraphs = personal_info.split("\n\n", 2)[:2]
            st.markdown("### About Me")
            st.write("\n\n".join(first_paragraphs))
    with col_animation:
        hero_animation = load_lottiefile(str(HOME_ANIMATIONS["hero"]))
        if hero_animation:
            st_lottie(hero_animation)

    st.divider()
    col_projects_left, col_projects_right = st.columns([2, 1])
    with col_projects_left:
        st.markdown(
            """
            ##### 🛠️ What You’ll Find Here
            * Interactive data exploration with dynamic filtering.
            * Ready-to-use visualisations powered by Plotly.
            * AI-assisted insights for both SQL queries and charts.
            * Automated profiling via YData Profiling.
            """
        )
    with col_projects_right:
        animation = load_lottiefile(str(HOME_ANIMATIONS["projects_left"]))
        if animation:
            st_lottie(animation)
        animation = load_lottiefile(str(HOME_ANIMATIONS["projects_right"]))
        if animation:
            st_lottie(animation, height=300)

    st.divider()
    col_future, col_future_anim = st.columns([2, 1])
    with col_future:
        st.markdown(
            """
            ##### 🔮 Future Work
            * Adding code export for graphs and DataFrame transformations.
            * Introducing query-based filtering.
            * Expanding error handling and observability.
            """
        )
    with col_future_anim:
        future_animation = load_lottiefile(str(HOME_ANIMATIONS["future"]))
        if future_animation:
            st_lottie(future_animation, height=150)

    st.divider()
    col_contact, col_contact_anim = st.columns([2, 1])
    with col_contact:
        st.markdown(
            """
            ##### 📞 Get in Touch
            * Connect with me on [LinkedIn](https://www.linkedin.com/in/samsonthedatascientist/)
            * Explore more on [GitHub](https://github.com/samsontands)
            * Email me at `samsontands@gmail.com`
            """
        )
    with col_contact_anim:
        contact_animation = load_lottiefile(str(HOME_ANIMATIONS["contact"]))
        if contact_animation:
            st_lottie(contact_animation, height=150)

    st.divider()
    st.markdown(
        """
        **If you find this project useful, please consider starring the GitHub repository and sharing it with your network.**

        **[`GitHub Repo Link >`](https://github.com/samsontands)**
        """
    )


def run() -> None:
    configure_environment()
    st.set_page_config(page_title="Samson Data Viewer", page_icon="📊", layout="wide")
    ensure_state_defaults(st.session_state)

    page = render_sidebar()

    if page != MenuPage.ASK_AI:
        st.title("**📋 Samson Data Viewer**", anchor=False)
        st.caption("**Made by Samson with AI❤️**")

    curr_filtered_df = pd.DataFrame()
    filter_log = ""
    if page in DATA_PAGES:
        curr_filtered_df, filter_log = render_data_filters()

    if page == MenuPage.HOME:
        render_home()
    elif page == MenuPage.DATAFRAME:
        render_dataframe_page(curr_filtered_df, filter_log)
    elif page == MenuPage.STATISTICS:
        render_statistics_page(curr_filtered_df, filter_log)
    elif page == MenuPage.GRAPHER:
        render_grapher_page(st, curr_filtered_df)
    elif page == MenuPage.PYGWALKER:
        render_pygwalker_page(curr_filtered_df)
    elif page == MenuPage.ASK_AI:
        filtered_df = st.session_state.get("filtered_df", pd.DataFrame())
        render_ai_assistant_page(filtered_df)
    elif page == MenuPage.PROJECTS:
        render_projects_page()
    elif page == MenuPage.ASK_ME_ANYTHING:
        render_ask_me_anything_page()
    elif page == MenuPage.PROFILING:
        show_eda_tool(st)


__all__ = ["run"]
