# CapitalFlow AI — Smart Traffic Volume Prediction Dashboard

**Graduation Project — AI & Data Science Internship at ACUD**

|                 |                                                              |
| --------------- | ------------------------------------------------------------ |
| Trainee Name    | Omar Ahmed Ramadan Ramadan                                   |
| Major / Program | Artificial Intelligence and Data Science Technology Program |
| University      | El Sewedy University of Technology                           |
| Company         | ACUD (Administrative Capital For Urban Development)          |
| Training Topic  | AI & Data Science Internship                                 |

An interactive Machine Learning and Deep Learning dashboard that predicts road traffic volume for Egypt's New Administrative Capital. This is the graduation project of the internship, built to bring together everything covered across the Machine Learning and Deep Learning phases of the program.

## Overview

CapitalFlow AI predicts how busy a specific road in the New Administrative Capital will be, given a trip (origin and destination), a date and time, and weather conditions. It combines three trained models — Random Forest, Gradient Boosting, and an Artificial Neural Network (ANN) — into an ensemble prediction, and wraps the whole thing in a full trip-planning dashboard: route selection, a Faster Road Advisor and a Road Closure Advisor, traffic analytics, and model performance comparisons.

The road network is split into gateway highways (the main axes connecting the New Capital to Greater Cairo) and internal districts (Government District, Financial District, R3/R5/R7, etc). Based on the origin and destination picked, every trip is automatically classified into one of four categories:

- **Entering the City** — from outside (a gateway) into an internal district
- **Exiting the City** — from an internal district out to a gateway
- **Inside the City** — from one internal district to another
- **Passing Through** — from one gateway to another, transiting through the Capital without stopping inside it

The underlying dataset is a synthetic but realistic hourly traffic dataset (January 2025 – March 2026) built around the real geography of the New Capital, Cairo-region weather patterns, Egyptian public holidays, and Friday–Saturday weekend logic.

## Features

- **Trip Planner** — pick an origin and destination from the New Capital's gateway highways and internal districts; the app resolves the correct road and automatically classifies the trip as Entering, Exiting, Inside the City, or Passing Through
- **Date & Time Selection** — pick any date/time within the dataset's range to get a time-aware prediction
- **Weather Input** — temperature, rainfall, cloud coverage, holiday, and weather condition, all used as live model inputs
- **Faster Road Advisor** — even without reporting anything, the app checks comparable roads under the same conditions and proactively suggests a faster one whenever it would meaningfully cut the trip time (at least 3 minutes saved)
- **Road Closure Advisor** — explicitly report a road as closed or heavily jammed, and the app recommends the least busy alternative among comparable roads, with an estimated extra delay, and confirms when a reported closure doesn't actually affect your selected trip
- **Ensemble Prediction** — averages Random Forest, Gradient Boosting, and ANN predictions for a more robust traffic volume estimate
- **Prediction Dashboard** — traffic volume estimate, a congestion status badge (LOW / MODERATE / HIGH / CRITICAL, based on predicted vehicles/hour), and an estimated extra wait time for the selected trip
- **Traffic Analytics** — traffic distribution, rush-hour breakdown, and historical trend charts for the selected road
- **Model Performance Panel** — compares all trained models (Random Forest, ANN, Gradient Boosting, Linear Regression, LSTM) on MAE, RMSE, and R²
- **Branded UI** — custom high-contrast theme with the ACUD logo integrated into the sidebar

## Model Performance

Five models were trained and compared during development; the three strongest were kept in the deployed ensemble (Random Forest, Gradient Boosting, ANN):

| Model             | MAE    | RMSE   | R²   |
| ----------------- | ------ | ------ | ----- |
| Random Forest     | 66.59  | 102.83 | 0.976 |
| ANN               | 90.36  | 131.72 | 0.961 |
| Gradient Boosting | 139.67 | 194.98 | 0.915 |
| Linear Regression | 331.99 | 409.44 | 0.626 |
| LSTM              | 332.50 | 454.64 | 0.539 |

## Tech Stack

- **Machine Learning:** scikit-learn (Random Forest, Gradient Boosting, Linear Regression), joblib
- **Deep Learning:** TensorFlow/Keras (ANN, LSTM)
- **Data Processing:** pandas, NumPy
- **Visualization:** Plotly
- **Web App:** Streamlit

## Project Structure

```
CapitalFlow_AI_final/
├── app.py                              # Streamlit dashboard application
├── code.ipynb                          # Data preparation, EDA, and model training notebook
├── New_Capital_Roads_Traffic_Volume.csv # Hourly traffic dataset (Jan 2025 – Mar 2026)
├── model_results.csv                   # Comparison metrics for all trained models
├── preprocessor.pkl                    # Fitted sklearn preprocessing pipeline
├── rf_model.pkl                        # Trained Random Forest model
├── gb_model.pkl                        # Trained Gradient Boosting model
├── ann_model.weights.h5                # Trained ANN weights
├── assets/
│   └── acud_logo.png                   # ACUD branding for the sidebar
├── requirements.txt                    # Python dependencies
└── README.md
```

## Installation

You have two options: run the app yourself from the terminal, or just open it directly from the live link with no setup at all.

```bash
git clone https://github.com/OmarAhmedRamadan07/CapitalFlow.git
cd CapitalFlow
pip install -r requirements.txt
```

## Usage

### Option 1 — Run the web app from the terminal

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, set up your trip, time, and weather, and click **Predict Traffic**.

### Option 2 — Open it directly from the link (no source code needed)

You don't have to clone the repo or install anything at all. The app is already deployed and ready to use straight from your browser:

https://capitalflow.streamlit.app/

### Run the notebook

Open `code.ipynb` to walk through data preparation, exploratory analysis, feature engineering, and the training/comparison of all five models.

## How It Works

1. **Data Preparation** — hourly traffic records covering every gateway and internal road, enriched with weather (temperature, rain, cloud cover, weather condition) and calendar context (holidays, weekends).
2. **Preprocessing** — a fitted scikit-learn preprocessing pipeline (`preprocessor.pkl`) encodes categorical fields (road, weather, holiday) and scales numerical ones, producing the exact feature format the trained models expect.
3. **Model Training** — five models were trained and compared: Random Forest, Gradient Boosting, Linear Regression, an ANN, and an LSTM (see performance table above).
4. **Ensemble Prediction** — at inference time, the app runs the preprocessed input through Random Forest, Gradient Boosting, and the ANN, then averages the three predictions for the final traffic volume estimate.
5. **Faster Road Advisor** — for every prediction, the app also checks the other comparable roads (same category — gateway vs. internal/feeder) under the exact same date, time, and weather, and proactively flags one of them if it comes out at least 3 minutes faster, even if nothing was reported as blocked.
6. **Road Closure Advisor** — when a road is explicitly marked as closed or jammed on the Closure tab, the app re-runs the fast sklearn models (Random Forest + Gradient Boosting) on every comparable candidate road under the same conditions, recommends whichever one comes out least busy, and estimates the extra delay caused by the closure. If the reported closure doesn't actually affect the selected trip, the app confirms that too.
7. **Dashboard** — the resulting prediction feeds a set of visual components: a congestion status indicator, estimated wait time, historical trend chart for the road, rush-hour distribution, and the live model performance comparison.

## Notes

- The three deployed models (`rf_model.pkl`, `gb_model.pkl`, `ann_model.weights.h5`) and the preprocessor are pre-trained; they are loaded as-is and are not retrained at runtime.
- Date selection in the app is limited to the dataset's covered range (January 2025 – March 2026).
- The dataset is synthetic but modeled on the real road network and general climate patterns of the New Administrative Capital.

## Acknowledgments

Built as the graduation project of the AI & Data Science Internship at ACUD (Administrative Capital for Urban Development), CET191 — Internship I, El Sewedy University of Technology.

## License

This project is provided for educational and portfolio purposes.
