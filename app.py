import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_extras.grid import grid
import streamlit_antd_components as sac
from streamlit_card import card
from streamlit_lottie import st_lottie
import traceback
from wordcloud import WordCloud
import pygwalker as pyg
import os
import json
from datetime import datetime
import requests
from ydata_profiling import ProfileReport
import io

# --- Additions for OpenAI + .env Ask CSV ---
from typing import Optional
from dotenv import load_dotenv

# Prefer new OpenAI SDK; fallback to legacy
try:
    from openai import OpenAI  # >=1.0
    _NEW_OPENAI_SDK = True
except Exception:
    import openai  # legacy
    _NEW_OPENAI_SDK = False

# Load .env (won’t override real env vars)
load_dotenv(override=False)

def _get_openai_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("openai_api_key", None)
    except Exception:
        return None

_openai_client = None
def _ensure_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = _get_openai_api_key()
        if not api_key:
            st.error("OpenAI API key not found. Set OPENAI_API_KEY in .env or 'openai_api_key' in Streamlit secrets.")
            st.stop()
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def generate_gpt_response(gpt_input: str, max_tokens: int) -> str:
    api_key = _get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not found. Set OPENAI_API_KEY in .env or 'openai_api_key' in Streamlit secrets.")
        st.stop()

    if _NEW_OPENAI_SDK:
        client = _ensure_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role":"user","content":gpt_input}],
        )
        return resp.choices[0].message.content.strip()
    else:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role":"user","content":gpt_input}],
        )
        return resp.choices[0].message['content'].strip()

def extract_code(gpt_response: str) -> str:
    # Pull code from ``` blocks; strip language hints like python/py/sql
    if "```" in gpt_response:
        m = re.search(r'```(.*?)```', gpt_response, re.DOTALL)
        if m:
            code = m.group(1)
            code = re.sub(r'^\s*(python|py|sql)\s*\n', '', code, flags=re.IGNORECASE)
            return code
    return gpt_response



# Guarantee a usable DataFrame handle across pages
if "curr_filtered_df" not in st.session_state:
    st.session_state.curr_filtered_df = pd.DataFrame()

os.environ['SKETCH_MAX_COLUMNS'] = '50'
st.set_page_config(
    page_title="Samson Data Viewer",
    page_icon="📊",
    layout="wide"
)

# ===== Default dataset config =====
DEFAULT_CSV_NAME = "sample_sales_data.csv"
# Public repo raw URL example:
# https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/sample_sales_data.csv
DEFAULT_CSV_URL  = "https://raw.githubusercontent.com/samsontands/portfolio3/main/sample_sales_data.csv"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_github_csv(url: str, token: str | None = None) -> pd.DataFrame:
    """
    Load a CSV from a GitHub raw URL.
    If token is provided (private repo), an authenticated request is used.
    """
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        # requests is already imported in your file
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.warning(f"Could not load default CSV from GitHub: {e}")
        return pd.DataFrame()


# Force Sketch to use its hosted endpoint (no OpenAI)
os.environ["SKETCH_USE_REMOTE_LAMBDAPROMPT"] = "True"

# Nuke anything that might push Sketch to OpenAI/HF
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("LAMBDAPROMPT_BACKEND", None)
os.environ.pop("LAMBDAPROMPT_OPENAI_MODEL", None)


import sketch
    



@st.cache_resource
def load_personal_info():
    with open('config/personal_info.txt', 'r') as f:
        return f.read()

@st.cache_resource(show_spinner = 0, experimental_allow_widgets=True)
def sidebar_animation(date):
    st_lottie(load_lottiefile("lottie_files/Animation - 1694990107205.json"))

def convert_df(df, index = False):
    return df.to_csv(index = index).encode('utf-8')

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def show_eda_tool():
    st.title('Data Profiling with YData Profiling')
    
    if st.session_state.select_df:
        df = st.session_state.filtered_df
        st.write(df)
        
        if st.button("Generate Profiling Report"):
            with st.spinner('Generating profiling report...'):
                profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
                
                # Generate the report as a string
                report_html = profile.to_html()
                
            st.success('Report generated successfully!')
            
            # Provide a download button for the HTML file
            st.download_button(
                label="Download Profiling Report",
                data=report_html,
                file_name="profiling_report.html",
                mime="text/html"
            )
    else:
        st.warning("Please select a dataframe from the sidebar first.")

def get_groq_response(prompt, system_prompt, personal_info):
    import requests
    import os
    import json
    import streamlit as st

    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("GROQ_API_KEY is missing in Streamlit secrets.")
        return ""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": f"{system_prompt} {personal_info}"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.2
    }

    try:
        # Use json= so requests sets the header and handles serialization
        resp = requests.post(url, headers=headers, json=payload, timeout=30)

        # If non-200, try to show API error details
        if not resp.ok:
            try:
                err = resp.json()
            except ValueError:
                err = {"error": {"message": resp.text}}
            msg = err.get("error", {}).get("message", f"HTTP {resp.status_code}")
            st.error(f"GROQ API returned an error: {msg}")
            # Optional: show debug info
            st.caption(f"Debug: status={resp.status_code}, body={err}")
            return ""

        data = resp.json()

        # Handle unexpected shapes gracefully
        if "choices" in data and data["choices"]:
            content = (
                data["choices"][0]
                .get("message", {})
                .get("content", "")
            )
            if not content:
                st.warning("Groq returned an empty message content.")
            return content

        # Sometimes API returns {'error': {...}} with 200 (rare, but guard anyway)
        if "error" in data:
            st.error(f"GROQ API error: {data['error'].get('message','Unknown error')}")
            st.caption(f"Debug payload: {data}")
            return ""

        # Fallback for unknown shapes
        st.error("Unexpected response from GROQ API (no 'choices').")
        st.caption(f"Debug payload: {data}")
        return ""

    except requests.exceptions.RequestException as e:
        st.error(f"Network error calling GROQ API: {e}")
        return ""

with st.sidebar:
    sidebar_animation(datetime.now().date())

    # --- File upload OR default from GitHub ---
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

    with st.expander("Upload files", expanded=not st.session_state.file_uploaded):
        st.session_state.files = st.file_uploader(
            "Upload files", type=["csv"], accept_multiple_files=True, label_visibility='collapsed'
        )

        st.session_state.file_name = {}
        loaded_any = False

        if st.session_state.files:
            # User uploads override default
            for i in range(len(st.session_state.files)):
                uploaded = st.session_state.files[i]
                st.session_state.file_name[uploaded.name] = i
                df_tmp = pd.read_csv(uploaded)
                df_tmp['Row_Number_'] = np.arange(0, len(df_tmp))
                st.session_state.files[i] = df_tmp
            loaded_any = True

        else:
            # If no uploads, try default GitHub CSV
            gh_token = st.secrets.get("GITHUB_TOKEN", None)  # only needed for private repos
            df_default = fetch_github_csv(DEFAULT_CSV_URL, token=gh_token)

            if not df_default.empty:
                df_default['Row_Number_'] = np.arange(0, len(df_default))
                st.session_state.files = [df_default]
                st.session_state.file_name = {DEFAULT_CSV_NAME: 0}
                loaded_any = True
            else:
                # No uploads and default failed
                st.session_state.files = []
                st.session_state.file_name = {}

        if loaded_any:
            st.session_state.select_df = selectbox(
                "**Select Dataframe**", st.session_state.file_name.keys(), no_selection_label=None
            )
            st.session_state.file_uploaded = True
        else:
            st.session_state.select_df = None
            st.session_state.filtered_df = pd.DataFrame()
            st.session_state.file_uploaded = False


    page = sac.menu([
    sac.MenuItem('Home', icon='house'),
    sac.MenuItem('DataFrame', icon='speedometer2'),
    sac.MenuItem('Statistics', icon='plus-slash-minus'),
    sac.MenuItem('Grapher', icon='graph-up'),
    # sac.MenuItem('Reshaper', icon='square-half'),
    sac.MenuItem('PygWalker', icon='plugin'),
    sac.MenuItem('Ask AI', icon='robot'),
    sac.MenuItem('My Projects', icon ='card-text'),
    sac.MenuItem('Ask Me Anything', icon='chat-dots'),
    sac.MenuItem('YData Profiling', icon='bar-chart-line')  # New menu item
    ], index=0, format_func='title', size='small', indent=15, open_index=None, open_all=True, return_index=True)

    st.markdown("""
    <h3 style='text-align: left; margin-bottom: 10px;'>Contact Information</h3>
    <style>
        .contact-info {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 5px 10px;
            align-items: center;
        }
        .contact-info img {
            width: 20px;
            height: 20px;
            vertical-align: middle;
        }
    </style>
    <div class="contact-info">
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXBob25lIj48cGF0aCBkPSJNMjIgMTYuOTJ2M2EyIDIgMCAwIDEtMi4xOCAyIDE5Ljc5IDE5Ljc5IDAgMCAxLTguNjMtMy4wNyAxOS41IDE5LjUgMCAwIDEtNi02IDE5Ljc5IDE5Ljc5IDAgMCAxLTMuMDctOC42N0EyIDIgMCAwIDEgNC4xMSAyaDNhMiAyIDAgMCAxIDIgMS43MiAxMi44NCAxMi44NCAwIDAgMCAuNyAyLjgxIDIgMiAwIDAgMS0uNDUgMi4xMUw4LjA5IDkuOTFhMTYgMTYgMCAwIDAgNiA2bDEuMjctMS4yN2EyIDIgMCAwIDEgMi4xMS0uNDUgMTIuODQgMTIuODQgMCAwIDAgMi44MS43QTIgMiAwIDAgMSAyMiAxNi45MnoiLz48L3N2Zz4=">
        <span>+6011-1122 1128</span>
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1haWwiPjxyZWN0IHdpZHRoPSIyMCIgaGVpZ2h0PSIxNiIgeD0iMiIgeT0iNCIgcng9IjIiLz48cGF0aCBkPSJtMjIgNy0xMCA3TDIgNyIvPjwvc3ZnPg==">
        <span>samsontands@gmail.com</span>
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1hcC1waW4iPjxwYXRoIGQ9Ik0yMCAxMGMwIDYtOCAxMi04IDEycy04LTYtOC0xMmE4IDggMCAwIDEgMTYgMFoiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEwIiByPSIzIi8+PC9zdmc+">
        <span>Kuala Lumpur, Malaysia</span>
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLWxpbmtlZGluIj48cGF0aCBkPSJNMTYgOGE2IDYgMCAwIDEgNiA2djdoLTR2LTdhMiAyIDAgMCAwLTItMiAyIDIgMCAwIDAtMiAydjdoLTR2LTdhNiA2IDAgMCAxIDYtNnoiLz48cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSIxMiIgeD0iMiIgeT0iOSIvPjxjaXJjbGUgY3g9IjQiIGN5PSI0IiByPSIyIi8+PC9zdmc+">
        <a href="https://www.linkedin.com/in/samsonthedatascientist/">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
@st.cache_resource(show_spinner = 0, experimental_allow_widgets=True)
def home(date):
    st.divider()
    col = st.columns([5, 1])
    with col[0].container():
        st.markdown('''##### Hi, I am Samson Tan Jia Sheng 👋
                    
#### A Data Scientist From Malaysia\n**I am passionate about Data Analysis, Data Visualization, Machine Learning, and AI advancements.**''')

        personal_info = load_personal_info()
        st.markdown("### About Me")
        st.write(personal_info.split('\n\n')[0])  # Display the first paragraph of your personal info

    with col[1].container():
        st_lottie(load_lottiefile("lottie_files/Animation - 1694988603751.json"))

    st.divider()

    col = st.columns([2, 1])
    with col[0].container():
        st.markdown('''##### :film_projector: About the Project
**This Pandas DataFrame Viewer is a powerful tool for data analysis and visualization.**
* **Perform quick and efficient data analysis on your datasets**
* **Visualize data using various chart types**
* **Leverage AI for intelligent data insights**
* **User-friendly interface for both beginners and advanced users**
* **Incorporates libraries like Streamlit, Pandas, Plotly, and more for robust functionality**

**If you find this project useful, please consider starring the GitHub repository and sharing it with your network.**

**[`GitHub Repo Link >`](https://github.com/samsontands)**
    ''')


    with col[1].container():
        st_lottie(load_lottiefile("lottie_files/Animation - 1694988937837.json"))
        st_lottie(load_lottiefile("lottie_files/Animation - 1694989926620.json"), height = 300)

    st.divider()

    col1 = st.columns([2, 1])

    with col1[0].container():
        st.markdown('''
    ##### 🔮 Future Work

    * **Adding Code Export for graphs and for changes in dataframe**
    * **Adding Query based filtering**
    * **More Error Handling**
    ''')
    with col1[1].container():
        st_lottie(load_lottiefile("lottie_files/Animation - 1694991370591.json"), height = 150)
    st.divider()
    col2 = st.columns([2, 1])
    with col2[0].container():
        st.markdown('''
        ##### 📞 Contact with me

        * **Connect with me on [`LinkedIn>`](https://www.linkedin.com/in/samsonthedatascientist/)**
        * **My Github Profile [`Github>`](https://github.com/samsontands)**
        * **Mail me on `samsontands@gmail.com`**
        ''')
    with col2[1].container():
        st_lottie(load_lottiefile("lottie_files/Animation - 1694990540946.json"), height = 150)

if page == 0:
    st.title("**📋 Samson Data Viewer**", anchor = False)
    st.caption("**Made by Samson with AI❤️**")
    home(datetime.now().date())
elif page != 6:
    st.title("**📋 Samson Data Viewer**", anchor = False)
    st.caption("**Made by Samson with AI❤️**")
    log = ''
    with st.expander(label = '**Filters**'):
        if st.session_state.select_df:
            try:
                st.session_state.filtered_df = st.session_state.files[st.session_state.file_name[st.session_state.select_df]]
                typess = ['int64', 'float64', 'str', 'bool', 'object', 'timestamp']
                columns_to_show_df = st.data_editor(
                    pd.DataFrame({
                        "Column Name": st.session_state.filtered_df.drop('Row_Number_', axis=1).columns.to_list(),
                        "Show?": True,
                        "Convert Type": st.session_state.filtered_df.drop('Row_Number_', axis=1).dtypes.astype(str)
                    }),
                    column_config={"Convert Type": st.column_config.SelectboxColumn("Convert Type", options=typess, required=True, default=5)},
                    num_rows="fixed", hide_index=True, disabled=["Columns"], height=250, use_container_width=True
                )
                for i in range(0, columns_to_show_df.shape[0]):
                    if columns_to_show_df["Convert Type"][i] == 'timestamp':
                        st.session_state.filtered_df[columns_to_show_df["Column Name"][i]] = pd.to_datetime(
                            st.session_state.filtered_df[columns_to_show_df["Column Name"][i]]
                        )
                    else:
                        st.session_state.filtered_df[columns_to_show_df["Column Name"][i]] = (
                            st.session_state.filtered_df[columns_to_show_df["Column Name"][i]].astype(
                                columns_to_show_df["Convert Type"][i]
                            )
                        )
                st.caption("**:red[Note:] Date / Time column will always be converted to Timestamp**")
                st.session_state.filtered_df = dataframe_explorer(st.session_state.filtered_df, case=False)
                st.session_state.filtered_df.drop('Row_Number_', axis=1, inplace=True)
                curr_filtered_df = st.session_state.filtered_df[
                    columns_to_show_df[columns_to_show_df['Show?'] == True]['Column Name'].to_list()
                ]
            except Exception:
                log = traceback.format_exc()
                curr_filtered_df = pd.DataFrame()

if page == 1:
    st.write("")
    if st.session_state.select_df:
        st.data_editor(curr_filtered_df, use_container_width = True, num_rows="dynamic", hide_index = False)
        st.caption("**:red[Note:] To delete rows, press delete button in keyboard after selecting rows**")
        st.markdown(f"**DataFrame Shape: {curr_filtered_df.shape[0]} x {curr_filtered_df.shape[1]}**")
        st.download_button(label="**Download Modified DataFrame as CSV**", data = convert_df(curr_filtered_df), file_name=f"{st.session_state.select_df}", mime='text/csv')
        st.subheader("**Console Log**", anchor = False)
        st.markdown(f'{log}')

elif page == 2:
    st.write("")
    if st.session_state.select_df:
        stats = curr_filtered_df.describe().copy().T
        stats['Unique'] = curr_filtered_df.apply(lambda x: len(x.unique()))
        st.dataframe(stats, use_container_width = True, hide_index = False)
        st.markdown(f"**DataFrame Shape: {curr_filtered_df.shape[0]} x {curr_filtered_df.shape[1]}**")
        st.download_button(label="**Download Statistics DataFrame as CSV**", data = convert_df(stats, index = True), file_name=f"stats_{st.session_state.select_df}", mime='text/csv')

elif page == 3:
    st.write("")
    grapher_tabs = sac.segmented(
    items=[
        sac.SegmentedItem(label='Scatter'),
        sac.SegmentedItem(label='Line'),
        sac.SegmentedItem(label='Bar'),
        sac.SegmentedItem(label='Histogram'),
        sac.SegmentedItem(label='Box'),
        sac.SegmentedItem(label='Violin'),
        sac.SegmentedItem(label='Scatter 3D'),
        sac.SegmentedItem(label='Heatmap'),
        sac.SegmentedItem(label='Contour'),
        sac.SegmentedItem(label='Pie'),
        sac.SegmentedItem(label='Splom'),
        sac.SegmentedItem(label='Candlestick'),
        sac.SegmentedItem(label='Word Cloud'),
    ], label=None, position='top', index=0, format_func='title', radius='md', size='md', align='center', direction='horizontal', grow=True, disabled=False, readonly=False, return_index=True)
    
    if st.session_state.select_df:
        colorscales = {'Plotly3': px.colors.sequential.Plotly3, 'Viridis': px.colors.sequential.Viridis, 'Cividis': px.colors.sequential.Cividis, 'Inferno': px.colors.sequential.Inferno, 'Magma': px.colors.sequential.Magma, 'Plasma': px.colors.sequential.Plasma, 'Turbo': px.colors.sequential.Turbo, 'Blackbody': px.colors.sequential.Blackbody, 'Bluered': px.colors.sequential.Bluered, 'Electric': px.colors.sequential.Electric, 'Jet': px.colors.sequential.Jet, 'Rainbow': px.colors.sequential.Rainbow, 'Blues': px.colors.sequential.Blues, 'BuGn': px.colors.sequential.BuGn, 'BuPu': px.colors.sequential.BuPu, 'GnBu': px.colors.sequential.GnBu, 'Greens': px.colors.sequential.Greens, 'Greys': px.colors.sequential.Greys, 'OrRd': px.colors.sequential.OrRd, 'Oranges': px.colors.sequential.Oranges, 'PuBu': px.colors.sequential.PuBu, 'PuBuGn': px.colors.sequential.PuBuGn, 'PuRd': px.colors.sequential.PuRd, 'Purples': px.colors.sequential.Purples, 'RdBu': px.colors.sequential.RdBu, 'RdPu': px.colors.sequential.RdPu, 'Reds': px.colors.sequential.Reds, 'YlOrBr': px.colors.sequential.YlOrBr, 'YlOrRd': px.colors.sequential.YlOrRd, 'turbid': px.colors.sequential.turbid, 'thermal': px.colors.sequential.thermal, 'haline': px.colors.sequential.haline, 'solar': px.colors.sequential.solar, 'ice': px.colors.sequential.ice, 'gray': px.colors.sequential.gray, 'deep': px.colors.sequential.deep, 'dense': px.colors.sequential.dense, 'algae': px.colors.sequential.algae, 'matter': px.colors.sequential.matter, 'speed': px.colors.sequential.speed, 'amp': px.colors.sequential.amp, 'tempo': px.colors.sequential.tempo, 'Burg': px.colors.sequential.Burg, 'Burgyl': px.colors.sequential.Burgyl, 'Redor': px.colors.sequential.Redor, 'Oryel': px.colors.sequential.Oryel, 'Peach': px.colors.sequential.Peach, 'Pinkyl': px.colors.sequential.Pinkyl, 'Mint': px.colors.sequential.Mint, 'Blugrn': px.colors.sequential.Blugrn, 'Darkmint': px.colors.sequential.Darkmint, 'Emrld': px.colors.sequential.Emrld, 'Aggrnyl': px.colors.sequential.Aggrnyl, 'Bluyl': px.colors.sequential.Bluyl, 'Teal': px.colors.sequential.Teal, 'Tealgrn': px.colors.sequential.Tealgrn, 'Purp': px.colors.sequential.Purp, 'Purpor': px.colors.sequential.Purpor, 'Sunset': px.colors.sequential.Sunset, 'Magenta': px.colors.sequential.Magenta, 'Sunsetdark': px.colors.sequential.Sunsetdark, 'Agsunset': px.colors.sequential.Agsunset, 'Brwnyl': px.colors.sequential.Brwnyl}
        if grapher_tabs == 0:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = selectbox('**Select y value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_1', no_selection_label = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_2', no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_3', no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_4', no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_5', no_selection_label = None)
                symbol = selectbox('**Select symbol value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_6', no_selection_label = None)
                size = selectbox('**Select size value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_1_7', no_selection_label = None)
                trendline = selectbox('**Select trendline**', ['ols', 'lowess'], key = 'grid_grapher_1_8', no_selection_label = None)
                marginal_x = selectbox('**Select marginal x**', ['histogram', 'rug', 'box', 'violin'], key = 'grid_grapher_1_9', no_selection_label = None)
                marginal_y = selectbox('**Select marginal y**', ['histogram', 'rug', 'box', 'violin'], key = 'grid_grapher_1_10', no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_1_11')
            with grid_grapher.expander("", expanded = True):
                try:
                    if y:
                        fig = px.scatter(data_frame = curr_filtered_df, x = x, y = y, color = color, symbol = symbol, size = size, trendline = trendline, marginal_x = marginal_x, marginal_y = marginal_y, facet_row = facet_row, facet_col = facet_col, height = 750, render_mode='auto', color_continuous_scale = colorscales[plot_color])
                        fig.update_layout(coloraxis = fig.layout.coloraxis)
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.scatter(height = 750, render_mode='auto'), use_container_width = True)
                    log = ''
                except Exception as e:
                    st.plotly_chart(px.scatter(height = 750, render_mode='auto'), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 1:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = st.multiselect('**Select y values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_2_1', default = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_2_2',no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_2_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_2_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_2_5',no_selection_label = None)
                aggregation = selectbox('**Select aggregation**', ['mean', 'median', 'min', 'max', 'sum'], key = 'grid_grapher_2_6',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_2_7')
            with grid_grapher.expander("", expanded = True):
                try:
                    line_plot_df = curr_filtered_df.copy()
                    key_cols_line = [val for val in [x, color, facet_row, facet_col] if val is not None]
                    if key_cols_line != []:
                        if aggregation is not None:
                            line_plot_df = curr_filtered_df.groupby(key_cols_line).agg(aggregation).reset_index()
                        else:
                            line_plot_df = curr_filtered_df.sort_values(key_cols_line)
                    if y:
                        fig = px.line(data_frame = line_plot_df, x = x, y = y, color = color, facet_row = facet_row, facet_col = facet_col, render_mode='auto', height = 750, color_discrete_sequence = colorscales[plot_color])
                        fig.update_traces(connectgaps=True)
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.line(height = 750, render_mode='auto'), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.line(height = 750, render_mode='auto'), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 2:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = st.multiselect('**Select y values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_3_1', default = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_3_2',no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_3_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_3_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_3_5',no_selection_label = None)
                aggregation = selectbox('**Select aggregation**', ['mean', 'median', 'min', 'max', 'sum'], key = 'grid_grapher_3_6',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_3_7')
                sort = selectbox('**Select sort type**', ['asc', 'desc'], key = 'grid_grapher_3_8',no_selection_label = None)
            with grid_grapher.expander("", expanded = True):
                try:
                    bar_plot_df = curr_filtered_df.copy()
                    key_cols_bar = [val for val in [x, color, facet_row, facet_col] if val is not None]
                    if key_cols_bar != []:
                        if aggregation is not None:
                            bar_plot_df = curr_filtered_df.groupby(key_cols_bar).agg(aggregation).reset_index()
                        else:
                            bar_plot_df = curr_filtered_df.sort_values(key_cols_bar)
                    if sort is not None:
                        if sort == 'asc':
                            bar_plot_df = bar_plot_df.sort_values(y, ascending=True)
                        else:
                            bar_plot_df = bar_plot_df.sort_values(y, ascending=False)
                    if y:
                        fig = px.bar(data_frame = bar_plot_df, x = x, y = y, color = color, facet_row = facet_row, facet_col = facet_col, height = 750, color_continuous_scale = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.bar(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.bar(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 3:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                x = st.multiselect('**Select x values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_4_1', default = None)
                color = selectbox('**Select color values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_4_2',no_selection_label = None)
                facet_row = selectbox('**Select facet row values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_4_3',no_selection_label = None)
                facet_col = selectbox('**Select facet col values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_4_4',no_selection_label = None)
                marginal = selectbox('**Select marginal**', ['rug', 'box', 'violin'], key = 'grid_grapher_4_5', no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_4_6')
                cumulative = st.checkbox('Cumulative ?', key = 'grid_grapher_4_7')
            with grid_grapher.expander("", expanded = True):
                try:
                    if x:
                        fig = px.histogram(data_frame = curr_filtered_df, x = x, color = color, facet_row = facet_row, facet_col = facet_col, marginal = marginal, cumulative = cumulative, height = 750, color_discrete_sequence = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.bar(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.bar(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 4:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = st.multiselect('**Select y values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_5_1', default = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_5_2',no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_5_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_5_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_5_5',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_5_6')
            with grid_grapher.expander("", expanded = True):
                try:
                    if y:
                        fig = px.box(data_frame = curr_filtered_df, x = x, y = y, color = color, facet_row = facet_row, facet_col = facet_col, height = 750, color_discrete_sequence = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.box(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.box(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 5:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = st.multiselect('**Select y values**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_6_1', default = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_6_2',no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_6_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_6_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_6_5',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_6_6')
            with grid_grapher.expander("", expanded = True):
                try:
                    if y:
                        fig = px.violin(data_frame = curr_filtered_df, x = x, y = y, color = color, facet_row = facet_row, facet_col = facet_col, height = 750, color_discrete_sequence = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.violin(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.violin(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 6:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = selectbox('**Select y value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_7_1', no_selection_label = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_7_2',no_selection_label = None)
                z = selectbox('**Select z value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_7_3',no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_7_4',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_7_5')
            with grid_grapher.expander("", expanded = True):
                try:
                    if y:
                        fig = px.scatter_3d(data_frame = curr_filtered_df, x = x, y = y, z = z, color = color, height = 750, color_discrete_sequence = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.bar(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.bar(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 7:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = selectbox('**Select y value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_8_1', no_selection_label = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_8_2',no_selection_label = None)
                z = selectbox('**Select z value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_8_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_8_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_8_5',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_8_6')
            with grid_grapher.expander("", expanded = True):
                try:
                    if y:
                        fig = px.density_heatmap(data_frame = curr_filtered_df, x = x, y = y, z = z, facet_row = facet_row, facet_col = facet_col, height = 750, color_continuous_scale = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.density_heatmap(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.density_heatmap(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 8:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                y = selectbox('**Select y value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_9_1', no_selection_label = None)
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_9_2',no_selection_label = None)
                z = selectbox('**Select z value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_9_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_9_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_9_5',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_9_6')
            with grid_grapher.expander("", expanded = True):
                try:
                    if y:
                        fig = px.density_contour(data_frame = curr_filtered_df, x = x, y = y, color = z, facet_row = facet_row, facet_col = facet_col, height = 750)
                        fig.update_traces(contours_coloring = 'fill', contours_showlabels = True, colorscale = plot_color)
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.density_contour(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.density_contour(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 9:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                name = selectbox('**Select name value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_10_1', no_selection_label = None)
                value = selectbox("**Select value's value**", curr_filtered_df.columns.to_list(), key = 'grid_grapher_10_2',no_selection_label = None)
                color = selectbox('**Select color value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_10_3',no_selection_label = None)
                facet_row = selectbox('**Select facet row value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_10_4',no_selection_label = None)
                facet_col = selectbox('**Select facet col value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_10_5',no_selection_label = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_10_6')
            with grid_grapher.expander("", expanded = True):
                try:
                    if name:
                        # if facet_row is not None or facet_col is not None:
                        #     raise NotImplementedError
                        fig = px.pie(data_frame = curr_filtered_df, names = name, values = value, color = color, facet_row = facet_row, facet_col = facet_col, height = 750, color_discrete_sequence = colorscales[plot_color])
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.pie(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.pie(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 10:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                dimensions = st.multiselect('**Select dimensions value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_11_1', default = None)
                color = selectbox('**Select color value (Column should be included as one of the dimension value)**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_11_2',no_selection_label = None)
                diag = st.selectbox("**Select Diagonal Plot**", ['scatter', 'histogram', 'box'], index = 1, key = 'grid_grapher_11_3')
                plot_color = st.selectbox("**Select Plot Color Map**", ['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu', 'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis'], index = 0, key = 'grid_grapher_11_4')
            
            with grid_grapher.expander("", expanded = True):
                try:
                    if dimensions:
                        # fig = px.scatter_matrix(data_frame = curr_filtered_df, dimensions = dimensions, color = color, height = 750, color_continuous_scale = colorscales[plot_color])
                        fig = ff.create_scatterplotmatrix(curr_filtered_df[dimensions], diag = diag, title = "", index = color, colormap = plot_color, height = 750)
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.bar(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.bar(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 11:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                x = selectbox('**Select x value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_12_1', no_selection_label = None)
                open = selectbox('**Select open value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_12_2',no_selection_label = None)
                high = selectbox('**Select high value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_12_3',no_selection_label = None)
                low = selectbox('**Select low value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_12_4',no_selection_label = None)
                close = selectbox('**Select close value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_12_5',no_selection_label = None)
            with grid_grapher.expander("", expanded = True):
                try:
                    if x and open and high and low and close:
                        fig = go.Figure(data=[go.Candlestick(x = curr_filtered_df[x], open = curr_filtered_df[open], high = curr_filtered_df[high], low = curr_filtered_df[low], close = curr_filtered_df[close])])
                        fig.update_layout(height=750)
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        st.plotly_chart(px.density_contour(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.density_contour(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif grapher_tabs == 12:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                words = st.multiselect('**Select words value**', curr_filtered_df.columns.to_list(), key = 'grid_grapher_13_1', default = None)
                plot_color = st.selectbox("**Select Plot Color Map**", colorscales.keys(), index = 0, key = 'grid_grapher_13_2')
            with grid_grapher.expander("", expanded = True):
                try:
                    if words:
                        if type(words) == str:
                            words = [words]
                        text = ' '.join(pd.concat([curr_filtered_df[x].dropna().astype(str) for x in words]))
                        wc = WordCloud(scale=2, collocations=False).generate(text)
                        st.plotly_chart(px.imshow(wc, color_continuous_scale = colorscales[plot_color]), height = 750, use_container_width = True)
                    else:
                        st.plotly_chart(px.bar(height = 750), use_container_width = True)
                except Exception as e:
                    st.plotly_chart(px.bar(height = 750), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

elif page == -1:
    st.write("")
    reshaper_tabs = sac.segmented(
    items=[
        sac.SegmentedItem(label='Pivot'),
        sac.SegmentedItem(label='Melt'),
        sac.SegmentedItem(label='Merge'),
        sac.SegmentedItem(label='Concat'),
        sac.SegmentedItem(label='Join')
    ], label=None, position='top', index=0, format_func='title', radius='md', size='md', align='center', direction='horizontal', grow=True, disabled=False, readonly=False, return_index=True)
    if st.session_state.select_df:  
        if reshaper_tabs == 0:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                index = st.multiselect('**Select index value**', curr_filtered_df.columns.to_list(), key = 'grid_reshaper_1_1', default = None)
                column = st.multiselect('**Select column value**', curr_filtered_df.columns.to_list(), key = 'grid_reshaper_1_2',default = None)
                value = st.multiselect("**Select value's value**", curr_filtered_df.columns.to_list(), key = 'grid_reshaper_1_3',default = None)
                aggfunc = st.selectbox('**Select aggfunc**', ['count','mean', 'median','mode','min','max','sum'], key = 'grid_reshaper_1_4', index = 1)
            with grid_grapher.expander("", expanded = True):
                try:
                    if index or column:
                        tmp = curr_filtered_df.pivot_table(index = index, columns = column, values = value, aggfunc = aggfunc).copy()
                        st.dataframe(tmp, height = 750, use_container_width = True)
                        st.markdown(f"**DataFrame Shape: {tmp.shape[0]} x {tmp.shape[1]}**")
                        st.download_button(label="**Download Modified DataFrame as CSV**", data = convert_df(tmp), file_name=f"Pivot_{st.session_state.select_df}", mime='text/csv')
                    else:
                        st.dataframe(pd.DataFrame(), use_container_width = True)
                except Exception as e:
                    st.dataframe(pd.DataFrame(), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif reshaper_tabs == 1:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            with grid_grapher.expander(label = 'Features', expanded = True):
                id_vars = st.multiselect('**Select id_vars value**', curr_filtered_df.columns.to_list(), key = 'grid_reshaper_2_1', default = None)
                value_vars = st.multiselect('**Select value_vars value**', curr_filtered_df.columns.to_list(), key = 'grid_reshaper_2_2', default = None)
            with grid_grapher.expander("", expanded = True):
                try:
                    if id_vars or value_vars:
                        tmp = curr_filtered_df.melt(id_vars = id_vars, value_vars = value_vars)
                        st.dataframe(tmp, height = 750, use_container_width = True)
                        st.markdown(f"**DataFrame Shape: {tmp.shape[0]} x {tmp.shape[1]}**")
                        st.download_button(label="**Download Modified DataFrame as CSV**", data = convert_df(tmp), file_name=f"Melt_{st.session_state.select_df}", mime='text/csv')
                    else:
                        st.dataframe(pd.DataFrame(), use_container_width = True)
                except Exception as e:
                    st.dataframe(pd.DataFrame(), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif reshaper_tabs == 2:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            other_dataframe = pd.DataFrame()
            with grid_grapher.expander(label = 'Features', expanded = True):
                other = selectbox("Select other Dataframe", list(filter(lambda x: x != st.session_state.select_df, st.session_state.file_name.keys())), key = 'grid_reshaper_3_1', no_selection_label = None)
                if other:
                    other_dataframe = st.session_state.files[st.session_state.file_name[other]].drop('Row_Number_', axis = 1)
                how = st.selectbox('**Select how**', ['inner', 'left', 'right', 'outer'], key = 'grid_reshaper_3_2', index = 0)
                left_on = st.multiselect('**Select left on values**', curr_filtered_df.columns.to_list(), key = 'grid_reshaper_3_3',default = None)
                right_on = st.multiselect('**Select right on values (Other DataFrame)**', other_dataframe.columns.to_list(), key = 'grid_reshaper_3_4',default = None)
                validate = selectbox('**Select validate**', ['one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'], key = 'grid_reshaper_3_5',no_selection_label = None)
            with grid_grapher.expander("", expanded = True):
                try:
                    if not(other_dataframe.empty) and left_on and right_on:
                        tmp = curr_filtered_df.merge(right = other_dataframe, how = how, left_on = left_on, right_on = right_on, validate = validate)
                        st.dataframe(tmp, height = 750, use_container_width = True)
                        st.markdown(f"**DataFrame Shape: {tmp.shape[0]} x {tmp.shape[1]}**")
                        st.download_button(label="**Download Modified DataFrame as CSV**", data = convert_df(tmp), file_name=f"Merged_{st.session_state.select_df}", mime='text/csv')
                    else:
                        st.dataframe(pd.DataFrame(), use_container_width = True)
                except Exception as e:
                    st.dataframe(pd.DataFrame(), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif reshaper_tabs == 3:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            other_dataframe = []
            with grid_grapher.expander(label = 'Features', expanded = True):
                other = st.multiselect("**Select other Dataframe**", list(filter(lambda x: x != st.session_state.select_df, st.session_state.file_name.keys())), key = 'grid_reshaper_4_1', default = None)
                if other:
                    other_dataframe = [st.session_state.files[st.session_state.file_name[df]].drop('Row_Number_', axis = 1) for df in other]
                axis = st.selectbox('**Select axis**', ['0 (rows)', '1 (columns)'], key = 'grid_reshaper_4_2')
                ignore_index = st.checkbox('Ignore Index ?', key = 'grid_reshaper_4_3')
            with grid_grapher.expander("", expanded = True):
                try:
                    if other_dataframe:
                        tmp = pd.concat([curr_filtered_df] + other_dataframe, axis = int(axis[0]), ignore_index = ignore_index)
                        st.dataframe(tmp, height = 750, use_container_width = True)
                        st.markdown(f"**DataFrame Shape: {tmp.shape[0]} x {tmp.shape[1]}**")
                        st.download_button(label="**Download Modified DataFrame as CSV**", data = convert_df(tmp), file_name=f"Concat_{st.session_state.select_df}", mime='text/csv')
                    else:
                        st.dataframe(pd.DataFrame(), use_container_width = True)
                except Exception as e:
                    st.dataframe(pd.DataFrame(), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

        elif reshaper_tabs == 4:
            grid_grapher = grid([1, 2], vertical_align="bottom")
            other_dataframe = pd.DataFrame()
            with grid_grapher.expander(label = 'Features', expanded = True):
                other = selectbox("Select other Dataframe", list(filter(lambda x: x != st.session_state.select_df, st.session_state.file_name.keys())), key = 'grid_reshaper_5_1', no_selection_label = None)
                if other:
                    other_dataframe = st.session_state.files[st.session_state.file_name[other]].drop('Row_Number_', axis = 1)
                how = st.selectbox('**Select how**', ['inner', 'left', 'right', 'outer'], key = 'grid_reshaper_5_2', index = 0)
                on = selectbox('**Select on values**', curr_filtered_df.columns.to_list(), key = 'grid_reshaper_5_3', no_selection_label = None)
                lsuffix = st.text_input("**Suffix to use from left frame's overlapping columns**", placeholder = "Enter lsuffix", key = 'grid_reshaper_5_4')
                rsuffix = st.text_input("**Suffix to use from right frame's overlapping columns**", placeholder = "Enter rsuffix", key = 'grid_reshaper_5_5')
                sort = st.checkbox('Sort ?', key = 'grid_reshaper_5_6')

            with grid_grapher.expander("", expanded = True):
                try:
                    if not(other_dataframe.empty):
                        tmp = curr_filtered_df.join(other_dataframe, how = how, on = on, lsuffix = lsuffix, rsuffix = rsuffix, sort = sort)
                        st.dataframe(tmp, height = 750, use_container_width = True)
                        st.markdown(f"**DataFrame Shape: {tmp.shape[0]} x {tmp.shape[1]}**")
                        st.download_button(label="**Download Modified DataFrame as CSV**", data = convert_df(tmp), file_name=f"Join_{st.session_state.select_df}", mime='text/csv')
                    else:
                        st.dataframe(pd.DataFrame(), use_container_width = True)
                except Exception as e:
                    st.dataframe(pd.DataFrame(), use_container_width = True)
                    log = traceback.format_exc()
            st.subheader("**Console Log**", anchor = False)
            st.markdown(f'{log}')

elif page == 4:
    if st.session_state.select_df:
        st.markdown("**Are you sure of proceeding to PygWalker interface?**")
        try:
            if st.button("Continue", key = 'PygWalker'):
                pyg.walk(curr_filtered_df, env = 'Streamlit', dark = 'media')
        except Exception as e:
            st.dataframe(pd.DataFrame(), use_container_width = True)
            log = traceback.format_exc()
        st.subheader("**Console Log**", anchor = False)
        st.markdown(f'{log}')


elif page == 5:
    # === Ask CSV (replacing Sketch Ask AI) ===
    st.title("Ask Your Data (AI) 🔎")
    st.caption("Type a question and I’ll generate a SQLite query against the filtered DataFrame — or ask me to create a Plotly chart.")

    if not st.session_state.select_df:
        st.warning("Please select a dataframe in the sidebar first.")
    else:
        # Use current filtered DF from your app
        df_ai = st.session_state.get("filtered_df", pd.DataFrame()).copy()
        if df_ai.empty:
            st.warning("Your filtered DataFrame is empty. Adjust filters or upload data.")
        else:
            # Clean columns for SQL friendliness
            df_ai = df_ai.reset_index(drop=True)
            df_ai.columns = df_ai.columns.str.replace(' ', '_', regex=False)

            tabs = st.tabs(["Ask (SQL)", "Create a chart"])
            # ------------------------------
            # Tab 1: Ask (SQL -> SQLite)
            # ------------------------------
            with tabs[0]:
                question = st.text_area(
                    "Ask a concise question. Example: What is the total sales in the USA in 2022?",
                    value="What is the total sales in the USA in 2022?"
                )
                if st.button("Run SQL"):
                    try:
                        # create in-memory SQLite from df_ai
                        conn = sqlite3.connect(":memory:")
                        table_name = "my_table"
                        df_ai.to_sql(table_name, conn, if_exists="replace", index=False)

                        cols = ", ".join(df_ai.columns.tolist())
                        prompt = (
                            "Write a SQLite query based on this question: {q} "
                            "The table name is my_table and the table has the following columns: {cols}. "
                            "Return only a SQL query and nothing else."
                        ).format(q=question, cols=cols)

                        sql = generate_gpt_response(prompt, max_tokens=250)
                        sql = extract_code(sql)

                        with st.expander("SQL used"):
                            st.code(sql, language="sql")

                        out = pd.read_sql_query(sql, conn)
                        if out.shape == (1, 1):
                            st.subheader(f"Answer: {out.iloc[0,0]}")
                        else:
                            st.subheader("Result")
                            st.dataframe(out, use_container_width=True)
                    except Exception:
                        st.error("Failed to run the generated SQL. Try rephrasing your question.")
                        st.code(traceback.format_exc())

            # ------------------------------
            # Tab 2: Create a chart
            # ------------------------------
            with tabs[1]:
                viz_req = st.text_area(
                    "Describe the chart you want. Example: Plot total sales by country and product category",
                    value="Plot total sales by country and product category"
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

                        code = generate_gpt_response(prompt, max_tokens=1500)
                        code = extract_code(code)

                        # Warn if generated code references columns we don’t have
                        missing = []
                        for m in re.findall(r'df\[\s*[\'"]([^\'"]+)[\'"]\s*\]', code):
                            if m not in df_ai.columns:
                                missing.append(m)
                        if missing:
                            st.warning(f"Columns not found in data: {sorted(set(missing))}")

                        # Replace common show() calls with Streamlit plotting
                        code = re.sub(r'(\b\w+\b)\.show\(\)',
                                      r"st.plotly_chart(\1, use_container_width=True)", code)
                        code = re.sub(r'plotly\.io\.show\((\b\w+\b)\)',
                                      r"st.plotly_chart(\1, use_container_width=True)", code)

                        with st.expander("Code used"):
                            st.code(code, language="python")

                        # Execute safely with our variables
                        _locals = {"st": st, "px": px, "go": go, "np": np, "pd": pd, "df": df_ai}
                        exec(code, {}, _locals)

                    except Exception:
                        st.error("Chart generation failed. Try a simpler description or different columns.")
                        st.code(traceback.format_exc())



elif page == 6:
    st.title('My Projects', anchor=False)

    # Custom CSS for better styling and clickable cards
    st.markdown("""
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
    .project-link {
        font-size: 0.9rem;
        color: #4CAF50;
        text-decoration: none;
    }
    .project-link:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Project data
    projects = [
        {
            "title": "Alliance Bank GPT",
            "description": "A webpage for AI that can answer simple questions and provide information.",
            "image": "https://images.prismic.io/codiste-website/08ac7396-b806-4550-b167-8814f6eb0fe2_What+is+the+difference+between+GPT_+GPT3%2C+GPT+3.5%2C+GPT+turbo+and+GPT-4.png?auto=compress,format",
            "url": "https://alliancegpt.streamlit.app/"
        },
        {
            "title": "File Transfer App",
            "description": "A webpage for temporary file transfer.",
            "image": "https://img.freepik.com/premium-photo/cloud-storage-icon-neon-element-black-background-3d-rendering-illustration_567294-1378.jpg?w=740",
            "url": "https://filecpdi.streamlit.app/"
        },
        {
            "title": "Depcreciation Analysis Demo",
            "description": "A website to showcase an analysis to calculate vehicle depreciation",
            "image": "https://static.vecteezy.com/system/resources/previews/005/735/523/original/thin-line-car-icons-set-in-black-background-universal-car-icon-to-use-in-web-and-mobile-ui-car-basic-ui-elements-set-free-vector.jpg",
            "url": "https://depreciationanalysis.streamlit.app/"
        },
        {
            "title": "Website Scraper POC",
            "description": "A website to showcase web scraper POC",
            "image": "https://miro.medium.com/v2/resize:fit:720/format:webp/1*nKwYuOo-zhF8eHocsR9WvA.png",
            "url": "https://scraperpoc.streamlit.app/"
        }
    ]

    # Sort projects alphabetically by title
    projects.sort(key=lambda x: x['title'])

    # Create a 2-column layout
    col1, col2 = st.columns(2)

    # Distribute projects across columns
    for i, project in enumerate(projects):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <a href="{project['url']}" target="_blank" style="text-decoration: none; color: inherit;">
                <div class="project-card">
                    <img src="{project['image']}" style="width:100%; border-radius:5px; margin-bottom:0.5rem;">
                    <div class="project-title">{project['title']}</div>
                    <div class="project-description">{project['description']}</div>
                </div>
            </a>
            """, unsafe_allow_html=True)

    # Add some spacing at the bottom
    st.markdown("<br>", unsafe_allow_html=True)

elif page == 7:  # Assuming the new menu item is at index 8
    st.title("Ask Me Anything")
    st.write("Ask a question to get a brief response about the creator's background, skills, or experience.")
    
    # Load necessary configuration files
    with open('config/personal_info.txt', 'r') as f:
        personal_info = f.read()

    with open('config/system_prompt.txt', 'r') as f:
        system_prompt = f.read()
    
    user_question = st.text_input("What would you like to know?")
    if user_question:
        with st.spinner('Getting a quick answer...'):
            response = get_groq_response(user_question, system_prompt, personal_info)
        st.write(response)
    st.caption("Note: Responses are kept brief. For more detailed information, please refer to other sections of the app.")
elif page == 8:  # Assuming YData Profiling is the 10th item (index 9) in your menu
    show_eda_tool()
