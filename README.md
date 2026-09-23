# 🚦 CapitalFlow AI

**By Omar Ahmed Ramadan**
🎓 Data Science & AI Technology Student — SUTech (El Sewedy University of Technology)
🏢 AI & Data Science Internship Project — ACUD (Administrative Capital for Urban Development)

---

An interactive 📊 Streamlit dashboard that predicts hour-by-hour road traffic
volume across Egypt's New Administrative Capital, and helps a driver pick
the best route — including live road-closure handling and a faster-route
advisor.

## ✨ What it does

- 🚗 **Traffic prediction** — pick an origin, a destination, a time, and the
  weather, and the app predicts the traffic volume (vehicles/hour) on the
  road that trip uses, plus an estimated extra wait time versus free flow.
- 🚧 **Road Closure Advisor** — report a road as blocked or heavily jammed.
  If it doesn't affect your trip, the app tells you so. If it does, it
  finds the least busy alternative and estimates how much time it saves.
- ⚡ **Faster Road Advisor** — even with no closures, if a different road
  would carry less predicted traffic for the same trip, the app flags it.
- 📈 **Traffic analytics** — historical charts (by road, by hour, by weather
  condition, etc.) built from the underlying dataset.
- 🏆 **Model performance** — a comparison table/chart of the trained models.
- 📉 **Traffic trend** — a time-series view of how volume moves over time.

## 🧠 How prediction works

Three models were trained on the historical dataset and are loaded
pre-fit at runtime (no retraining happens in the app):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| 🌳 Random Forest | 66.6 | 102.8 | 0.976 |
| 🧬 ANN (Keras) | 90.4 | 131.7 | 0.961 |
| 🚀 Gradient Boosting | 139.7 | 195.0 | 0.915 |

(Linear Regression and LSTM were also tried and are kept in
`model_results.csv` for comparison, but are not used for live prediction.)

The app blends the three deployed models' outputs into a single
prediction, then classifies it into a traffic status (e.g. Moderate,
Heavy) and converts it into an estimated wait time.

## 🗺️ Road network

- 🛣️ **Gateway roads** (into/out of the city): Cairo-Suez Road,
  Cairo-Ain Sokhna Road, Middle Ring Road, Regional Ring Road,
  Mohamed Bin Zayed Axis (North/South), Al-Amal Axis.
- 🏙️ **Internal districts**: CBD, Government District, Financial District,
  R3, R5, R7, Diplomatic Quarter, Green River Corridor.
- 🚸 **Side/feeder streets**: one lighter-traffic feeder road per internal
  district, used by the Road Closure Advisor for road-specific numbers.

Trip origins/destinations map to whichever road actually carries that
trip's traffic. A few newer locations (R5 District, all feeder streets)
don't have their own trained model category yet, so they're scored using
the closest trained road as a stand-in (see `DISPLAY_TO_MODEL_ROAD` in
`app.py`).

## ▶️ Running it locally

## 📝 Notes

- The ANN architecture is rebuilt in code and its weights are loaded
  from `ann_model.weights.h5` (rather than loading a full saved model),
  so the architecture in `app.py` must stay in sync with how the ANN was
  trained in `code.ipynb`.
- All UI text, labels, and code comments are in English by design, even
  though the app is used and discussed in Arabic day to day.

## 👤 Credits

Built by **Omar Ahmed Ramadan** — Data Science & AI Technology student,
SUTech (El Sewedy University of Technology) — as part of an AI & Data
Science internship at **ACUD**, Administrative Capital for Urban
Development ([acud.eg](https://acud.eg/)). 🏢🇪🇬
