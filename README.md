# Samson's Data Viewer

*A refreshed Streamlit workspace for data storytelling.*

## Description
Samson's Data Viewer is a Streamlit-powered workspace for rapid exploration, visualization, and storytelling with tabular data. The refreshed interface introduces a cohesive dark theme, guided chart builders, and quick access to personal portfolio links so collaborators can discover the person behind the project.

## Feature Highlights
- **Conversational analysis powered by Groq and Sketch** for natural-language insights.
- **Integrated YData Profiling** to generate shareable data quality reports in a single click.
- **Guided Plotly grapher** with contextual help and empty-state messaging across scatter, bar, heatmap, candlestick, word cloud, and more.
- **Data transformations and downloads** including filtering, reshaping, and CSV export.
- **Hero home page** with quick stats, "What's new" updates, and calls to action for portfolio and LinkedIn.

## What's New
- 🚀 Groq-powered conversations guide chart building and data wrangling.
- 🧠 Instant data audits through YData Profiling integration.
- 🎨 Streamlit theme overhaul with custom colors and typography for a cohesive feel.
- 🧭 Grapher tabs now include contextual hints and empty states so users know which fields to fill before a plot renders.

## Installation

### Prerequisites
- Python 3.7+
- pip

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/your-username/samsons-data-viewer.git
   cd samsons-data-viewer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Add your GROQ API key in Streamlit secrets: `.streamlit/secrets.toml`
   - (Optional) Add a GitHub token if loading private CSV files.

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload a CSV or use the bundled sample dataset from the sidebar.
3. Explore dataframe views, guided visualizations, Groq-assisted insights, and YData profiling reports.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Samson Tan – samsontands@gmail.com  
LinkedIn: [linkedin.com/in/samsonthedatascientist](https://www.linkedin.com/in/samsonthedatascientist/)
