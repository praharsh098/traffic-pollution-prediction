# Traffic & Vehicle Pollution Prediction

Predict city traffic and vehicular pollution levels using historical data and time-series forecasting.

[![CI](https://img.shields.io/github/actions/workflow/status/your-username/Traffic-&-Pollution-Prediction/ci.yml?branch=main)](https://github.com/your-username/Traffic-&-Pollution-Prediction/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

## Features
- 🎨 Interactive Streamlit dashboard
- 📊 Data cleaning & preprocessing
- ⏰ Time-based & lag features
- 🤖 Baseline linear regression model
- 📈 Interactive visualizations with Plotly
- 💾 Predictions saved to CSV
- 🔄 Real-time data upload support

## Project Structure
```
Traffic-Pollution-Prediction/
├── app.py                      # Streamlit app entry point
├── streamlit_config.toml       # Streamlit configuration
├── data/
│   ├── vehicle_emission.csv
│   └── delhi_traffic.csv
├── notebooks/
│   └── Traffic_Pollution_EDA_and_Modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── visualization.py
├── outputs/
│   ├── plots/
│   └── predictions.csv
├── requirements.txt
├── README.md
└── run_project.py
```

## Setup
- Python 3.10 recommended
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Data
Place the following files in `data/`:
- `vehicle_emission.csv`
- `delhi_traffic.csv`

Minimum required columns:
- `datetime` (ISO-like string or parseable date-time)
- Additional numeric columns used by your features/model (see `src/data_processing.py`, `src/feature_engineering.py`).

## Usage

### 🚀 Streamlit App (Recommended)

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Upload your own CSV files or use default data
- Interactive visualizations with Plotly
- Real-time data exploration
- Train models and view predictions
- Download results as CSV

### 📓 Jupyter Notebook

Execute the full pipeline via the notebook:

```bash
python run_project.py
```

Outputs:
- Plots in `outputs/plots/`
- Predictions in `outputs/predictions.csv`

### 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## Development
- Linting/tests are not enforced, but CI runs a smoke import of modules and dependencies on each push/PR to `main`.
- Update `requirements.txt` if you add dependencies.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
