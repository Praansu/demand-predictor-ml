# Parking Demand Predictor 📈

A machine learning project that predicts parking occupancy based on historical data and time-based features. Built to understand how well simple models can forecast parking demand — and where they fall short.

## What This Does

Takes historical parking occupancy data (hourly, with timestamps) and predicts future occupancy using:
- **Linear Regression** — baseline, interpretable
- **Random Forest** — captures non-linear patterns
- **XGBoost** — gradient boosting for best performance

Evaluates with MAE, RMSE, and R². Visualizes predictions vs actuals.

## Why I Built This

Wanted to understand: *Can we actually predict parking demand well enough to be useful?* Turns out — yes for general patterns, no for special events. The model learns daily/weekly cycles easily but fails on holidays, concerts, weather spikes. That's the interesting part.

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Generate sample data (or use your own CSV)
python src/generate_data.py --days 365 --output data/parking.csv

# 3. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 4. Train models
python src/train.py --data data/parking.csv --models all

# 5. Evaluate
python src/evaluate.py --model models/xgb_model.pkl --data data/parking.csv
```

## Project Structure

```
demand-predictor-ml/
├── data/                 # CSV data (generated or your own)
├── models/               # Saved model pickles
├── notebooks/
│   └── 01_eda.ipynb     # Exploratory data analysis
├── src/
│   ├── generate_data.py  # Synthetic data generator
│   ├── preprocess.py     # Feature engineering
│   ├── train.py          # Model training
│   └── evaluate.py       # Evaluation + plots
├── requirements.txt
└── README.md
```

## What I Learned

- **Feature engineering matters more than model choice.** Hour-of-day, day-of-week, is-weekend, is-holiday — these 4 features get you 80% of the way there.
- **Random Forest ≈ XGBoost** for this problem. Linear regression is surprisingly decent (R² ~0.72) because the signal is mostly periodic.
- **Special events break everything.** A concert or holiday spikes occupancy 3x — the model has no way to know unless you feed it event data.
- **MAE is more interpretable than RMSE.** "Off by 12 cars on average" means more to stakeholders than "RMSE 15.3".

## Model Performance (on synthetic 1-year data)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 14.2 | 18.7 | 0.72 |
| Random Forest | 9.8 | 13.1 | 0.86 |
| XGBoost | 9.1 | 12.4 | 0.88 |

*Your mileage will vary with real data — especially if you have event/weather features.*

## Tech Stack

- Python 3.10+
- pandas, numpy, scikit-learn, xgboost
- matplotlib, seaborn for viz
- joblib for model persistence

## Next Steps (If I Had More Time)

- [ ] Add weather API integration (OpenWeatherMap)
- [ ] Add event calendar scraping (Ticketmaster, local venues)
- [ ] Try Prophet or NeuralProphet for time-series specific modeling
- [ ] Build a simple FastAPI endpoint for real-time predictions
- [ ] Add drift detection for model monitoring

## License

MIT — use it, break it, improve it.