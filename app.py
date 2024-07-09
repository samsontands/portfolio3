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
import sketch
import os
import json
from datetime import datetime
import requests
from ydata_profiling import ProfileReport
import io

os.environ['SKETCH_MAX_COLUMNS'] = '50'
st.set_page_config(
    page_title="Samson Data Viewer",
    page_icon="📊",
    layout="wide"
)

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
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768", 
        "messages": [
            {"role": "system", "content": f"{system_prompt} {personal_info}"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['message']['content']

with st.sidebar:
    sidebar_animation(datetime.now().date())
    page = sac.menu([
    sac.MenuItem('Home', icon='house'),
    sac.MenuItem('DataFrame', icon='speedometer2'),
    sac.MenuItem('Statistics', icon='plus-slash-minus'),
    sac.MenuItem('Grapher', icon='graph-up'),
    sac.MenuItem('Reshaper', icon='square-half'),
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
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXBob25lIj48cGF0aCBkPSJNMjIgMTYuOTJ2M2EyIDIgMCAwIDEtMi4xOCAyIDE5Ljc5IDE5Ljc5IDAgMCAxLTguNjMtMy4wNyAxOS41IDE5LjUgMCAwIDEtNi0gNiAxOS43OSAxOS43OSAwIDAgMS0zLjA3LTguNjdBMiAyIDAgMCAxIDQuMTEgMmgzYTIgMiAwIDAgMSAyIDEuNzIgMTIuODQgMTIuODQgMCAwIDAgLjcgMi44MSAyIDIgMCAwIDEtLjQ1IDIuMTFMOC4wOSA5LjkxYTE2IDE2IDAgMCAwIDYgNmwxLjI3LTEuMjdhMiAyIDAgMCAxIDIuMTEtLjQ1IDEyLjg0IDEyLjg0IDAgMCAwIDIuODEuN0EyIDIgMCAwIDEgMjIgMTYuOTJ6Ii8+PC9zdmc+">
        <span>+6011-1122 1128</span>
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1haWwiPjxyZWN0IHdpZHRoPSIyMCIgaGVpZ2h0PSIxNiIgeD0iMiIgeT0iNCIgcng9IjIiLz48cGF0aCBkPSJtMjIgNy0xMCA3TDIgNyIvPjwvc3ZnPg==">
        <span>samsontands@gmail.com</span>
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1hcC1waW4iPjxwYXRoIGQ9Ik0yMCAxMGMwIDYtOCAxMi04IDEycy04LTYtOC0xMmE4IDggMCAwIDEgMTYgMFoiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEwIiByPSIzIi8+PC9zdmc+">
        <span>Kuala Lumpur, Malaysia</span>
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLWxpbmsiPjxwYXRoIGQ9Ik0xMCA4LjJsNC44LTQuOGEzLjk5OSAzLjk5OSAwIDAgMSA1LjY1NyA1LjY1N0wxNS44IDE0YTQgNCAwIDAgMS01LjY1NyAwIi8+PHBhdGggZD0ibTE0IDE1LjgtNC44IDQuOGEzLjk5OSAzLjk5OSAwIDEgMS01LjY1Ny01LjY1N0w4LjIgMTBhNCA0IDAgMCAxIDUuNjU3IDAiLz48L3N2Zz4=">
        <a href="https://www.linkedin.com/in/samsonthedatascientist/">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(label = '**Upload files**', expanded = False):
        st.session_state.files = st.file_uploader("Upload files", type = ["csv"], accept_multiple_files = True, label_visibility = 'collapsed')
        if st.session_state.files:
            st.session_state.file_name = {}
            for i in range(0, len(st.session_state.files)):
                st.session_state.file_name[st.session_state.files[i].name] = i
                if 'csv' in st.session_state.files[i].name or 'CSV' in st.session_state.files[i].name:
                    st.session_state.files[i] = pd.read_csv(st.session_state.files[i])
                    st.session_state.files[i]['Row_Number_'] = np.arange(0, len(st.session_state.files[i]))
            st.session_state.select_df = selectbox("**Select Dataframe**", st.session_state.file_name.keys(), no_selection_label = None)
        else:
            st.session_state.select_df = None
            st.session_state.filtered_df = pd.DataFrame()

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
elif page != 7:
    st.title("**📋 Samson Data Viewer**", anchor = False)
    st.caption("**Made by Samson with AI❤️**")
    log = ''
    with st.expander(label = '**Filters**'):
        if st.session_state.select_df:
            try:
                st.session_state.filtered_df = st.session_state.files[st.session_state.file_name[st.session_state.select_df]]
                typess = ['int64', 'float64', 'str', 'bool', 'object', 'timestamp']
                columns_to_show_df = st.data_editor(pd.DataFrame({"Column Name": st.session_state.filtered_df.drop('Row_Number_', axis = 1).columns.to_list(), "Show?": True, "Convert Type": st.session_state.filtered_df.drop('Row_Number_', axis = 1).dtypes.astype(str)}), column_config = {"Convert Type": st.column_config.SelectboxColumn("Convert Type", options = typess, required=True, default = 5)}, num_rows="fixed", hide_index = True, disabled = ["Columns"], height = 250, use_container_width = True)
                for i in range(0, columns_to_show_df.shape[0]):
                    if columns_to_show_df["Convert Type"][i] == 'timestamp':
                        st.session_state.filtered_df[columns_to_show_df["Column Name"][i]] = pd.to_datetime(st.session_state.filtered_df[columns_to_show_df["Column Name"][i]])
                    else:
                        st.session_state.filtered_df[columns_to_show_df["Column Name"][i]] = st.session_state.filtered_df[columns_to_show_df["Column Name"][i]].astype(columns_to_show_df["Convert Type"][i])
                st.caption("**:red[Note:] Date / Time column will always be converted to Timestamp**")
                st.session_state.filtered_df = dataframe_explorer(st.session_state.filtered_df, case=False)
                st.session_state.filtered_df.drop('Row_Number_', axis = 1, inplace = True)
            except:
                log = traceback.format_exc()
            curr_filtered_df = st.session_state.filtered_df[columns_to_show_df[columns_to_show_df['Show?'] == True]['Column Name'].to_list()]

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

elif page == 4:
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

elif page == 5:
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


elif page == 6:
    if st.session_state.select_df:
        preference_ai = st.radio("**Select your Preference**", options = ["**Ask about the selected Dataframe**", "**Ask how to perform actions on selected Dataframe**"], horizontal = True)
        prompt = st.text_area("Enter Promt", placeholder = "Enter your promt", label_visibility="collapsed")
        proceed_ai = st.button("Continue", key = 'ask_ai')
        with st.expander("**AI says**", expanded = True):
            st.divider()
            try:
                if preference_ai == "**Ask about the selected Dataframe**" and prompt and proceed_ai:
                    st.markdown(curr_filtered_df.sketch.ask(prompt, call_display=False))
                elif preference_ai == "**Ask how to perform actions on selected Dataframe**" and prompt and proceed_ai:
                    st.markdown(curr_filtered_df.sketch.howto(prompt, call_display=False))
            except Exception as e:
                st.dataframe(pd.DataFrame(), use_container_width = True)
                log = traceback.format_exc()
        st.subheader("**Console Log**", anchor = False)
        st.markdown(f'{log}')
elif page == 7:
    st.title('My Projects', anchor=False)
    card_grid = grid(3, vertical_align="center")
    
    with card_grid.container():
        card(
            title="Pandas Dataframe Viewer",
            text="A website for quick data analysis and visualization of your dataset with AI",
            image="https://user-images.githubusercontent.com/66067910/266804437-e9572603-7982-4b19-9732-18a079d48f5b.png",
            url="https://github.com/sumit10300203/Pandas-DataFrame-Viewer", 
            on_click=lambda: None
        )
        
    # Add your project details here
    with card_grid.container():
        card(
            title="Alliance Bank GPT",
            text="A brief description of your project goes here.",
            image="https://images.prismic.io/codiste-website/08ac7396-b806-4550-b167-8814f6eb0fe2_What+is+the+difference+between+GPT_+GPT3%2C+GPT+3.5%2C+GPT+turbo+and+GPT-4.png?auto=compress,format",
            url="https://github.com/samsontands/alliancegpt",
            on_click=lambda: None
        )
elif page == 8:  # Assuming the new menu item is at index 8
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
elif page == 9:  # Assuming YData Profiling is the 10th item (index 9) in your menu
    show_eda_tool()

