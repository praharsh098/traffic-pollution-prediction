# Traffic & Vehicle Pollution Prediction

Predict city traffic and vehicular pollution levels using historical data and time-series forecasting.

## Features
- Data cleaning & preprocessing
- Time-based & lag features
- Baseline linear regression model
- Visualization of trends
- Predictions saved to CSV

## Project Structure
```
Traffic-Pollution-Prediction/
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

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place datasets in `data/`: `vehicle_emission.csv` and `delhi_traffic.csv` (both must include `datetime`)
3. Run the project: `python run_project.py` (executes the notebook end-to-end)
4. View results in `outputs/`
