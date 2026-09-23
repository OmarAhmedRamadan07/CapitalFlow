# ============================================================
# CAPITALFLOW AI
# Smart Traffic Volume Prediction Dashboard
# New Administrative Capital Road Network
# ============================================================

import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, date




def sample_weather(rng):
    idx = rng.choice(len(weather_pool), p=weather_weights)
    return weather_pool[idx]


@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.pkl")



# ============================================================
# SECTION: PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CapitalFlow AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SECTION: HTML RENDER HELPER
# ============================================================
# Streamlit's markdown renderer treats any line indented 4+ spaces
# as a preformatted code block, so it prints raw "<div ...>" text
# instead of rendering it. Our HTML snippets are written with
# Python-style indentation for readability, so every line is
# stripped of leading whitespace here before being handed to
# st.markdown — this is what makes the cards, badges, and hero
# section render as actual styled boxes instead of literal text.

def render_html(html):
    lines = [line.strip() for line in html.strip("\n").split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ============================================================
# SECTION: ROAD NETWORK MAP
# ============================================================
# The New Capital road network is split into two groups:
#
#   GATEWAY roads: the main highways/axes that connect the New
#   Administrative Capital to Greater Cairo and the rest of the
#   country. Trips on these roads are either "Entering the City"
#   or "Exiting the City" depending on direction of travel.
#
#   INTERNAL roads: the key corridors and districts inside the
#   New Capital itself (Government District, Financial District,
#   R3, R5, R7, etc). Trips on these roads stay "Inside the City".
# ============================================================

GATEWAY_ROADS = [
    "Cairo-Suez Road",
    "Cairo-Ain Sokhna Road",
    "Middle Ring Road",
    "Regional Ring Road",
    "Mohamed Bin Zayed Axis (North)",
    "Mohamed Bin Zayed Axis (South)",
    "Al-Amal Axis",
]

INTERNAL_ROADS = [
    "Central Business District (CBD)",
    "Government District (Ministries Area)",
    "Financial District (Bank Area)",
    "R3 District",
    "R7 District",
    "R5 District",
    "Diplomatic Quarter",
    "Green River Corridor",
]

# Local feeder / side streets that branch off each internal
# district. They now have their own rows in the dataset (lighter
# traffic than their parent district's main road), which is what
# gives the Road Closure Advisor real, road-specific numbers to
# work with instead of one generic "closed road" estimate.
SIDE_ROADS = [
    "R3 Feeder Road",
    "R7 Feeder Road",
    "R5 Feeder Road",
    "CBD Feeder Road",
    "Government District Feeder Road",
    "Financial District Feeder Road",
    "Diplomatic Quarter Feeder Road",
    "Green River Feeder Road",
]

# Road types offered in the Closure tab, grouped the way a driver
# would actually think about them.
ROAD_TYPE_GROUPS = {
    "🛣️ Gateway Road (into / out of the city)": GATEWAY_ROADS,
    "🏙️ Internal District": INTERNAL_ROADS,
    "🚸 Side / Feeder Street": SIDE_ROADS,
}

# The already-trained models were fit on a fixed set of "road_name"
# categories (things like "C3 District", "C7 District", the seven
# gateway highways, and the five original internal districts).
# R5 District and every side/feeder street are newer additions with
# their own real rows in the dataset for charts and history, but no
# distinct training data yet — so for live predictions they're
# scored using their closest trained relative as a stand-in.
DISPLAY_TO_MODEL_ROAD = {
    "R3 District": "C3 District",
    "R7 District": "C7 District",
    "R5 District": "C3 District",

    "R3 Feeder Road": "C3 District",
    "R7 Feeder Road": "C7 District",
    "R5 Feeder Road": "C3 District",
    "CBD Feeder Road": "Central Business District (CBD)",
    "Government District Feeder Road": "Government District (Ministries Area)",
    "Financial District Feeder Road": "Financial District (Bank Area)",
    "Diplomatic Quarter Feeder Road": "Diplomatic Quarter",
    "Green River Feeder Road": "Green River Corridor",
}


def to_model_road(display_road):
    """Translate a road name shown in the UI to the category name
    the trained models were fitted on."""
    return DISPLAY_TO_MODEL_ROAD.get(display_road, display_road)


# Locations a person can pick as their origin or destination.
# "Outside the City" points map to the gateway road that serves
# that direction of travel; internal points map to the internal
# corridor that carries that district's traffic.

LOCATION_TO_ROAD = {

    # Outside-the-city locations, tied to their gateway highway
    "Greater Cairo (via Cairo-Suez Road)": "Cairo-Suez Road",
    "Ain Sokhna / Suez Coast (via Cairo-Ain Sokhna Road)": "Cairo-Ain Sokhna Road",
    "New Cairo / Ring Road Area (via Middle Ring Road)": "Middle Ring Road",
    "10th of Ramadan / Eastern Cairo (via Regional Ring Road)": "Regional Ring Road",
    "Northern Districts (via Mohamed Bin Zayed Axis North)": "Mohamed Bin Zayed Axis (North)",
    "Southern Districts (via Mohamed Bin Zayed Axis South)": "Mohamed Bin Zayed Axis (South)",
    "Mostakbal City / Al-Amal Corridor (via Al-Amal Axis)": "Al-Amal Axis",

    # Inside-the-city locations, tied to their internal corridor
    "Central Business District (CBD)": "Central Business District (CBD)",
    "Government District (Ministries Area)": "Government District (Ministries Area)",
    "Financial District (Bank Area)": "Financial District (Bank Area)",
    "R3 District": "R3 District",
    "R7 District": "R7 District",
    "R5 District": "R5 District",
    "Diplomatic Quarter": "Diplomatic Quarter",
    "Green River Corridor": "Green River Corridor",
}

OUTSIDE_LOCATIONS = [
    loc for loc, road in LOCATION_TO_ROAD.items() if road in GATEWAY_ROADS
]

INSIDE_LOCATIONS = [
    loc for loc, road in LOCATION_TO_ROAD.items() if road in INTERNAL_ROADS
]

ALL_LOCATIONS = OUTSIDE_LOCATIONS + INSIDE_LOCATIONS


def resolve_trip(origin, destination):
    """
    Work out which road carries a trip, and whether that trip
    is entering the city, exiting the city, or staying inside it.
    """

    origin_is_outside = origin in OUTSIDE_LOCATIONS
    destination_is_outside = destination in OUTSIDE_LOCATIONS

    if origin_is_outside and not destination_is_outside:
        # Coming in from outside toward an internal district
        direction = "Entering the City"
        road = LOCATION_TO_ROAD[origin]

    elif destination_is_outside and not origin_is_outside:
        # Leaving an internal district toward the outside world
        direction = "Exiting the City"
        road = LOCATION_TO_ROAD[destination]

    elif origin_is_outside and destination_is_outside:
        # Passing through / between two gateway corridors
        direction = "Passing Through"
        road = LOCATION_TO_ROAD[origin]

    else:
        # Both origin and destination are inside the city
        direction = "Inside the City"
        road = LOCATION_TO_ROAD[destination]

    return road, direction


# ============================================================
# SECTION: CUSTOM CSS (bright, high-contrast theme)
# ============================================================

render_html(
    """
<style>

    /* Main Background - light, airy, and easy to read */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(255, 159, 28, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6, 214, 160, 0.12),
                transparent 30%
            ),
            #F7F9FC;
        color: #1B2432;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #FFFFFF 0%,
                #F1F5F9 100%
            );
        border-right: 1px solid rgba(15,23,42,0.08);
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #1B2432;
    }

    /* Header */
    .hero {
        padding: 30px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(255,159,28,0.18),
                rgba(6,214,160,0.18)
            );
        border: 1px solid rgba(15,23,42,0.08);
        box-shadow:
            0 20px 60px rgba(15,23,42,0.08);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
        color: #1B2432;
    }

    .hero-subtitle {
        color: #475569;
        font-size: 17px;
    }

    /* Cards */
    .card {
        background:
            rgba(255,255,255,0.85);
        border:
            1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 22px;
        box-shadow:
            0 15px 40px rgba(15,23,42,0.06);
        backdrop-filter: blur(12px);
    }

    .metric-title {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #1B2432;
    }

    .metric-small {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Prediction */
    .prediction-card {
        background:
            linear-gradient(
                135deg,
                rgba(255,159,28,0.22),
                rgba(6,214,160,0.20)
            );
        border: 1px solid rgba(15,23,42,0.10);
        border-radius: 26px;
        padding: 30px;
        text-align: center;
        box-shadow:
            0 25px 70px rgba(15,23,42,0.10);
    }

    .prediction-label {
        color: #334155;
        font-size: 16px;
    }

    .prediction-value {
        font-size: 62px;
        font-weight: 900;
        margin: 10px 0;
        color: #1B2432;
    }

    .prediction-unit {
        color: #64748B;
        font-size: 15px;
    }

    /* Trip direction badge */
    .direction-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 14px;
        margin-top: 6px;
        background: rgba(255,159,28,0.15);
        color: #C2540A;
        border: 1px solid rgba(255,159,28,0.35);
    }

    /* Section titles */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #1B2432;
    }

    /* Status */
    .status {
        padding: 10px 18px;
        border-radius: 50px;
        display: inline-block;
        font-weight: 700;
        margin-top: 8px;
    }

    .status-low {
        background: rgba(6,214,160,0.18);
        color: #0A8F6B;
    }

    .status-medium {
        background: rgba(255,209,102,0.28);
        color: #A8710A;
    }

    .status-high {
        background: rgba(255,159,28,0.22);
        color: #C2540A;
    }

    .status-critical {
        background: rgba(239,71,111,0.18);
        color: #C81E4F;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 13px;
        font-weight: 700;
        font-size: 16px;
        background:
            linear-gradient(
                135deg,
                #FF9F1C,
                #06D6A0
            );
        color: #10241D;
        transition: 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 10px 30px rgba(255,159,28,0.30);
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: rgba(15,23,42,0.08);
        margin: 25px 0;
    }

    /* Prediction Context mini-cards (replaces st.metric, which
       cuts long text off with "...") */
    .context-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 14px;
        padding: 14px 12px;
        min-height: 78px;
    }

    .context-label {
        color: #64748B;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .context-value {
        color: #1B2432;
        font-size: 17px;
        font-weight: 700;
        line-height: 1.3;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: break-word;
    }

    /* Sidebar - give the trip form more room so location names
       don't need to be cut off */
    section[data-testid="stSidebar"] {
        width: 400px !important;
    }

    /* Selectbox (closed box + dropdown list) - let long location
       names wrap onto a second line instead of being truncated
       with "..." */
    div[data-baseweb="select"] > div {
        height: auto !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] * {
        white-space: normal !important;
        text-overflow: unset !important;
        overflow: visible !important;
        word-break: break-word;
    }

    /* Dropdown list items - consistent spacing for every row,
       whether its text wraps to one line or several */
    div[data-baseweb="popover"] li {
        white-space: normal !important;
        height: auto !important;
        min-height: 0 !important;
        box-sizing: border-box !important;
        padding: 12px 14px !important;
        margin: 0 !important;
        line-height: 1.45 !important;
        display: flex !important;
        align-items: flex-start !important;
        border-bottom: 1px solid rgba(15,23,42,0.06);
    }

    div[data-baseweb="popover"] li > * {
        white-space: normal !important;
        height: auto !important;
        line-height: 1.45 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Alternative-route warning card */
    .alert-card {
        background: rgba(239, 71, 111, 0.08);
        border: 1px solid rgba(239, 71, 111, 0.35);
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 14px;
    }

    .alert-title {
        color: #EF476F;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 6px;
    }

    .alert-body {
        color: #1B2432;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Small "before / after" cards shown under Est. Wait Time
       when a road closure is reported */
    .mini-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 10px;
    }

    .mini-card-alert {
        background: rgba(239, 71, 111, 0.08);
        border: 1px solid rgba(239, 71, 111, 0.35);
    }

    .mini-label {
        color: #64748B;
        font-size: 12px;
        margin-bottom: 2px;
    }

    .mini-value {
        color: #1B2432;
        font-size: 20px;
        font-weight: 700;
    }

    .mini-value-alert {
        color: #EF476F;
    }

</style>
"""
)


# ============================================================
# SECTION: LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # Load dataset
    data = pd.read_csv(
        "New_Capital_Roads_Traffic_Volume.csv"
    )

    # Clean column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Convert datetime
    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce"
    )

    # Remove invalid dates
    data = data.dropna(
        subset=["date_time"]
    ).copy()

    # Sort chronologically
    data = data.sort_values(
        "date_time"
    ).reset_index(drop=True)

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    # Year
    data["year"] = (
        data["date_time"].dt.year
    )

    # Month
    data["month"] = (
        data["date_time"].dt.month
    )

    # Day
    data["day"] = (
        data["date_time"].dt.day
    )

    # Hour
    data["hour"] = (
        data["date_time"].dt.hour
    )

    # Day of week
    data["day_of_week"] = (
        data["date_time"].dt.dayofweek
    )

    # Weekend (Egypt weekend: Friday-Saturday)
    data["is_weekend"] = (
        data["day_of_week"].isin([4, 5])
    ).astype(int)

    # Rush hour
    data["is_rush_hour"] = (
        (
            (data["hour"] >= 7) &
            (data["hour"] <= 9)
        )
        |
        (
            (data["hour"] >= 16) &
            (data["hour"] <= 19)
        )
    ).astype(int)

    return data


# ============================================================
# SECTION: BUILD ANN ARCHITECTURE
# ============================================================
# The ANN architecture is rebuilt here in code, then its trained
# weights are loaded from a weights-only file. This avoids Keras
# full-model deserialization, which can break across different
# Keras/TensorFlow versions installed on different machines.

def build_ann_model(input_dim):

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    return model


# ============================================================
# SECTION: LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        "preprocessor.pkl"
    )

    rf_model = joblib.load(
        "rf_model.pkl"
    )

    gb_model = joblib.load(
        "gb_model.pkl"
    )

    # Number of columns the preprocessor outputs, used to build
    # the ANN's input layer with the correct shape.
    ann_input_dim = len(
        preprocessor.get_feature_names_out()
    )

    ann_model = build_ann_model(
        ann_input_dim
    )

    ann_model.load_weights(
        "ann_model.weights.h5"
    )

    return (
        preprocessor,
        rf_model,
        gb_model,
        ann_model
    )


# ============================================================
# SECTION: LOAD MODEL RESULTS
# ============================================================

@st.cache_data
def load_results():

    try:

        results = pd.read_csv(
            "model_results.csv"
        )

        return results

    except:

        return pd.DataFrame()


# ============================================================
# SECTION: INITIALIZE
# ============================================================

df = load_data()

(
    preprocessor,
    rf_model,
    gb_model,
    ann_model
) = load_models()

model_results = load_results()


# ============================================================
# SECTION: HEADER
# ============================================================

render_html(
    """
<div class="hero">

    <div class="hero-title">
        🚦 CapitalFlow AI
    </div>

    <div class="hero-subtitle">
        Predicting road traffic across the New Administrative Capital, hour by hour
        <br>
        Random Forest • Gradient Boosting • Neural Network
    </div>

</div>
"""
)


# ============================================================
# SECTION: SIDEBAR - TRIP PLANNER CONTROLS
# ============================================================

with st.sidebar:

    _logo_path = Path(__file__).parent / "assets" / "acud_logo.png"

    if _logo_path.exists():
        _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
        render_html(
f"""
<div style="text-align:center; padding: 4px 0 16px 0;">
    <a href="https://acud.eg/" target="_blank" style="display: block;">
        <img src="data:image/png;base64,{_logo_b64}"
                style="width: 50%; height: auto; display: block; margin: 0 auto;" />
    </a>
</div>
"""
        )

    st.markdown("## 🎛️ Prediction Control")

    st.caption(
        "Set your trip, pick a time, and describe the weather — "
        "then run the prediction."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        tab_trip, tab_time, tab_weather, tab_closure = st.tabs(
            ["🗺️ Trip", "📅 Time", "🌦️ Weather", "🚧 Closure"]
        )

        # ----------------------------------------------------
        # TAB 1: TRIP
        # ----------------------------------------------------

        with tab_trip:

            st.markdown("**Where are you going?**")

            # Both selectboxes below explicitly recompute their own
            # `index` from whatever was already saved in
            # st.session_state, on every rerun (including the rerun
            # triggered by clicking "Predict Traffic"). This is what
            # keeps "From" and "To" showing your last pick instead of
            # snapping back to the first item in the list — we no
            # longer just trust Streamlit to carry the value forward
            # on its own, which is what was causing "To" to reset.

            origin_index = 0
            if (
                "origin_select" in st.session_state
                and st.session_state["origin_select"] in ALL_LOCATIONS
            ):
                origin_index = ALL_LOCATIONS.index(
                    st.session_state["origin_select"]
                )

            origin = st.selectbox(
                "🟢 From",
                ALL_LOCATIONS,
                index=origin_index,
                key="origin_select",
                help="The district or gateway road you're starting from."
            )

            destination_options = [
                loc for loc in ALL_LOCATIONS if loc != origin
            ]

            # If the previous "To" is still a valid choice (i.e. it's
            # not the same place as the current "From"), keep it
            # selected. Only fall back to the first option in the
            # rare case where the saved destination is no longer
            # valid (the user just set "From" to what used to be
            # their "To").
            destination_index = 0
            if (
                "destination_select" in st.session_state
                and st.session_state["destination_select"] in destination_options
            ):
                destination_index = destination_options.index(
                    st.session_state["destination_select"]
                )

            destination = st.selectbox(
                "🔴 To",
                destination_options,
                index=destination_index,
                key="destination_select",
                help="The district or gateway road you're heading to."
            )

        # ----------------------------------------------------
        # TAB 2: DATE & TIME
        # ----------------------------------------------------

        with tab_time:

            st.markdown("**When are you traveling?**")

            # The dataset now runs Jan 2025 through Mar 2026, so
            # the date picker is locked to that range.
            default_date = datetime.now().date()
            if not (date(2025, 1, 1) <= default_date <= date(2026, 3, 31)):
                default_date = date(2025, 1, 1)

            selected_date = st.date_input(
                "📅 Date",
                value=default_date,
                min_value=date(2025, 1, 1),
                max_value=date(2026, 3, 31)
            )

            selected_time = st.time_input(
                "🕐 Time",
                value=datetime.now().replace(
                    minute=0,
                    second=0,
                    microsecond=0
                ).time()
            )

        # ----------------------------------------------------
        # TAB 3: WEATHER
        # ----------------------------------------------------

        with tab_weather:

            st.markdown("**What's the weather like?**")

            temp = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-10.0,
                max_value=55.0,
                value=25.0,
                step=0.5
            )

            rain = st.number_input(
                "🌧️ Rain (1h, mm)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1
            )

            clouds = st.slider(
                "☁️ Cloud Coverage (%)",
                min_value=0,
                max_value=100,
                value=40
            )

            holiday_options = ["None"] + sorted(
                df["holiday"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            holiday_choice = st.selectbox(
                "🎉 Holiday",
                holiday_options
            )

            holiday = None if holiday_choice == "None" else holiday_choice

            weather_options = sorted(
                df["weather_main"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            weather_main = st.selectbox(
                "🌦️ Weather",
                weather_options
            )

            description_options = sorted(
                df["weather_description"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            weather_description = st.selectbox(
                "🌤️ Weather Description",
                description_options
            )

        # ----------------------------------------------------
        # TAB 4: ROAD CLOSURE
        # ----------------------------------------------------

        with tab_closure:

            st.markdown("**Is a road blocked or jammed right now?**")

            road_blocked = st.checkbox(
                "🚧 Report a closed / heavily jammed road",
                key="road_blocked_check"
            )

            road_type_options = list(ROAD_TYPE_GROUPS.keys())

            road_type_index = 0
            if (
                "road_type_select" in st.session_state
                and st.session_state["road_type_select"] in road_type_options
            ):
                road_type_index = road_type_options.index(
                    st.session_state["road_type_select"]
                )

            road_type_choice = st.selectbox(
                "Road type",
                road_type_options,
                index=road_type_index,
                key="road_type_select",
                help="Pick the kind of road first, then the "
                     "specific one below."
            )

            blocked_road_options = ROAD_TYPE_GROUPS[road_type_choice]

            blocked_road_index = 0
            if (
                "blocked_road_select" in st.session_state
                and st.session_state["blocked_road_select"] in blocked_road_options
            ):
                blocked_road_index = blocked_road_options.index(
                    st.session_state["blocked_road_select"]
                )

            blocked_road = st.selectbox(
                "Which road?",
                blocked_road_options,
                index=blocked_road_index,
                key="blocked_road_select",
                help="Only used if the checkbox above is ticked. "
                     "We'll suggest the least busy alternative and "
                     "estimate the extra delay."
            )

        st.markdown("")

        predict_button = st.form_submit_button(
            "🚀 Predict Traffic",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Route preview — updates once the form above is submitted
    # --------------------------------------------------------

    road, direction = resolve_trip(origin, destination)

    st.markdown("---")

    st.markdown("**Selected route**")

    render_html(
    f"""
        <div class="direction-badge">
            {direction}
        </div>
        """
)

    st.markdown(f"**Road used:** {road}")


# ============================================================
# SECTION: BUILD MODEL INPUT
# ============================================================

selected_datetime = pd.Timestamp(
    datetime.combine(
        selected_date,
        selected_time
    )
)

year = selected_datetime.year
month = selected_datetime.month
day = selected_datetime.day
hour = selected_datetime.hour
day_of_week = selected_datetime.dayofweek

is_weekend = int(
    day_of_week in (4, 5)
)

is_rush_hour = int(
    (
        7 <= hour <= 9
    )
    or
    (
        16 <= hour <= 19
    )
)


input_data = pd.DataFrame({

    "temp": [temp],

    "rain_1h": [rain],

    "clouds_all": [clouds],

    "holiday": [holiday],

    "weather_main": [weather_main],

    "weather_description": [
        weather_description
    ],

    "road_name": [to_model_road(road)],

    "year": [year],

    "month": [month],

    "day": [day],

    "hour": [hour],

    "day_of_week": [day_of_week],

    "is_weekend": [is_weekend],

    "is_rush_hour": [is_rush_hour]
})


# ============================================================
# SECTION: DEFAULT PREDICTION STATE
# ============================================================

prediction = None
ann_prediction = None
rf_prediction = None
gb_prediction = None


# ============================================================
# SECTION: RUN PREDICTION
# ============================================================

if predict_button:

    try:

        processed_input = preprocessor.transform(
            input_data
        )

        processed_input = np.asarray(
            processed_input,
            dtype=np.float32
        )

        # Random Forest
        rf_prediction = float(
            rf_model.predict(
                processed_input
            )[0]
        )

        # Gradient Boosting
        gb_prediction = float(
            gb_model.predict(
                processed_input
            )[0]
        )

        # ANN
        ann_prediction = float(
            ann_model.predict(
                processed_input,
                verbose=0
            )[0][0]
        )

        # Average of strong models
        prediction = np.mean([
            rf_prediction,
            gb_prediction,
            ann_prediction
        ])

        prediction = max(
            0,
            prediction
        )

        st.session_state["prediction"] = prediction
        st.session_state["rf_prediction"] = rf_prediction
        st.session_state["gb_prediction"] = gb_prediction
        st.session_state["ann_prediction"] = ann_prediction
        st.session_state["road"] = road
        st.session_state["direction"] = direction

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )


# ============================================================
# SECTION: RESTORE FROM SESSION STATE
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state[
        "prediction"
    ]

    rf_prediction = st.session_state[
        "rf_prediction"
    ]

    gb_prediction = st.session_state[
        "gb_prediction"
    ]

    ann_prediction = st.session_state[
        "ann_prediction"
    ]

    road = st.session_state.get(
        "road",
        road
    )

    direction = st.session_state.get(
        "direction",
        direction
    )


# ============================================================
# SECTION: TRAFFIC STATUS HELPER
# ============================================================

def traffic_status(value):

    if value < 1000:

        return (
            "LOW",
            "status-low"
        )

    elif value < 2000:

        return (
            "MODERATE",
            "status-medium"
        )

    elif value < 3200:

        return (
            "HIGH",
            "status-high"
        )

    else:

        return (
            "CRITICAL",
            "status-critical"
        )


# ============================================================
# SECTION: ESTIMATED WAIT TIME HELPER
# ============================================================
# Converts predicted traffic volume into an approximate extra
# delay, on top of free-flow driving time, so the prediction
# feels concrete rather than just a raw vehicles/hour number.

def estimate_wait_time(value):

    # Delay (minutes) at a few reference traffic volumes.
    volume_points = [0, 1000, 2000, 3200, 5000]
    delay_points = [0, 3, 8, 16, 30]

    delay_minutes = float(
        np.interp(value, volume_points, delay_points)
    )

    return round(delay_minutes)


# ============================================================
# SECTION: ALTERNATIVE ROUTE HELPER
# ============================================================
# When the selected road is congested, this checks the other
# roads of the same kind (gateway highways vs. internal
# districts) under the same date/time/weather conditions, and
# recommends whichever one the models expect to be least busy.

def predict_volume_for_road(road_display_name):

    try:

        candidate_input = input_data.copy()
        candidate_input["road_name"] = to_model_road(
            road_display_name
        )

        processed = preprocessor.transform(
            candidate_input
        )

        processed = np.asarray(
            processed,
            dtype=np.float32
        )

        # Only the fast sklearn models are used here. This function
        # gets called once per candidate road every time an
        # alternative is looked up, and the ANN (TensorFlow) call
        # has enough per-call overhead that doing it 7-15+ times a
        # run was what made the app feel slow / unresponsive.
        rf_p = float(
            rf_model.predict(processed)[0]
        )

        gb_p = float(
            gb_model.predict(processed)[0]
        )

        return max(
            0,
            float(np.mean([rf_p, gb_p]))
        )

    except Exception:

        return None


def find_alternative_road(current_road):

    if current_road in GATEWAY_ROADS:
        candidates = GATEWAY_ROADS
    else:
        # Internal districts and their side/feeder streets can
        # reasonably substitute for one another (e.g. a blocked
        # district road can be swapped for its own feeder street,
        # or vice versa).
        candidates = INTERNAL_ROADS + SIDE_ROADS

    # Roads that share the same trained category as the current
    # road (e.g. R5 District currently shares data with R3
    # District) would just repeat the same prediction, so they're
    # skipped as "alternatives".
    seen_model_roads = {to_model_road(current_road)}

    best_road = None
    best_volume = None

    for candidate in candidates:

        if candidate == current_road:
            continue

        model_name = to_model_road(candidate)

        if model_name in seen_model_roads:
            continue

        seen_model_roads.add(model_name)

        volume = predict_volume_for_road(candidate)

        if volume is None:
            continue

        if best_volume is None or volume < best_volume:
            best_volume = volume
            best_road = candidate

    return best_road, best_volume


# ============================================================
# SECTION: PREDICTION DASHBOARD
# ============================================================

if prediction is not None:

    status, status_class = traffic_status(
        prediction
    )

    wait_minutes = estimate_wait_time(
        prediction
    )

    # --------------------------------------------------------
    # Closure numbers, computed up front so both the small
    # "before / after" cards next to the prediction and the
    # alert card further down can reuse the same numbers.
    #
    # This only runs the full impact calculation when the
    # reported road is actually the one this trip uses — closing
    # a road you're not driving on shouldn't change your results.
    # --------------------------------------------------------

    closure_affects_trip = (
        road_blocked
        and blocked_road
        and blocked_road == road
    )

    closure_normal_wait = None
    closure_wait = None
    closure_alt_road = None
    closure_alt_volume = None
    closure_alt_wait = None
    closure_volume = None

    if closure_affects_trip:

        closure_alt_road, closure_alt_volume = find_alternative_road(
            blocked_road
        )

        closure_base_volume = predict_volume_for_road(blocked_road)

        if closure_alt_road is not None and closure_base_volume is not None:

            # A closure doesn't just remove that road's capacity —
            # its own traffic backs up into whatever's left of it
            # and the surrounding streets. This models that
            # back-up as a multiplier on the volume it would
            # normally be carrying, so the impact genuinely scales
            # with how busy that specific road already is.
            CLOSURE_BACKUP_MULTIPLIER = 2.5

            closure_volume = min(
                closure_base_volume * CLOSURE_BACKUP_MULTIPLIER,
                5000
            )

            closure_normal_wait = estimate_wait_time(closure_base_volume)
            closure_wait = estimate_wait_time(closure_volume)
            closure_alt_wait = estimate_wait_time(closure_alt_volume)

    # --------------------------------------------------------
    # Prediction + Metrics
    # --------------------------------------------------------

    col1, col5 = st.columns(
        [2, 1]
    )

    with col1:

        render_html(
    f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    Predicted Traffic Volume
                    <br>
                    {origin} → {destination}
                </div>

                <div class="prediction-value">
                    {prediction:,.0f}
                </div>

                <div class="prediction-unit">
                    vehicles / hour on {road}
                </div>

                <div class="status {status_class}">
                    {status}
                </div>

            </div>
            """
)

    with col5:

        render_html(
    f"""
            <div class="card">

                <div class="metric-title">
                    ⏱️ Est. Wait Time
                </div>

                <div class="metric-value">
                    +{wait_minutes} min
                </div>

                <div class="metric-small">
                    extra delay vs. free-flow
                </div>

            </div>
            """
)

        if closure_wait is not None:

            render_html(
    f"""
            <div class="mini-card">
                <div class="mini-label">
                    Via {closure_alt_road}
                </div>
                <div class="mini-value">
                    +{closure_alt_wait} min
                </div>
            </div>

            <div class="mini-card mini-card-alert">
                <div class="mini-label">
                    Time After Closure
                </div>
                <div class="mini-value mini-value-alert">
                    +{closure_wait} min
                </div>
            </div>
            """
)


    # --------------------------------------------------------
    # Faster Road Advisor — checks the other roads of the same
    # kind and speaks up whenever one of them is meaningfully
    # quicker, not just when traffic is officially bad.
    # --------------------------------------------------------

    alt_road, alt_volume = find_alternative_road(road)

    # Don't recommend a road that has itself been reported as
    # blocked / heavily jammed in the Closure tab — suggesting a
    # "faster" route that's actually closed would be misleading.
    alt_road_is_reported_blocked = (
        road_blocked
        and blocked_road
        and alt_road == blocked_road
    )

    if (
        alt_road is not None
        and alt_volume < prediction
        and not alt_road_is_reported_blocked
    ):

        alt_wait = estimate_wait_time(alt_volume)
        minutes_saved = wait_minutes - alt_wait

        # Only speak up if it's actually worth rerouting for.
        if minutes_saved >= 3:

            if status in ("HIGH", "CRITICAL"):
                alert_title = f"⚠️ Heavy traffic on {road}"
            else:
                alert_title = "💡 A faster road is available"

            render_html(
    f"""
            <div class="alert-card">

                <div class="alert-title">
                    {alert_title}
                </div>

                <div class="alert-body">
                    Try <b>{alt_road}</b> instead — the models expect
                    about {alt_volume:,.0f} vehicles/hour there
                    (vs {prediction:,.0f} on {road}), roughly
                    <b>{minutes_saved} min</b> faster.
                </div>

            </div>
            """
)

    st.markdown("")


    # --------------------------------------------------------
    # Road Closure Advisor (explicit "this road is blocked"
    # report from the Closure tab). Numbers were already computed
    # above so the mini cards next to the prediction match this
    # alert card exactly.
    # --------------------------------------------------------

    if road_blocked and blocked_road:

        if not closure_affects_trip:

            # The road they reported isn't the one their trip
            # actually uses, so nothing about their route changes.
            st.success(
                f"✅ {blocked_road} is reported blocked, but your "
                f"trip uses {road} — no impact on you, you're good "
                f"as you are."
            )

        elif closure_alt_road is not None and closure_wait is not None:

            minutes_saved = closure_wait - closure_alt_wait

            render_html(
    f"""
            <div class="alert-card">

                <div class="alert-title">
                    🚧 {blocked_road} is reported blocked / jammed
                </div>

                <div class="alert-body">
                    Staying on {blocked_road} means fighting roughly
                    <b>{closure_volume:,.0f} vehicles/hour</b> of
                    backed-up traffic — about
                    <b>+{closure_wait} min</b> of delay.
                    <br><br>
                    Go via <b>{closure_alt_road}</b> instead — expected
                    about {closure_alt_volume:,.0f} vehicles/hour there,
                    roughly a <b>+{closure_alt_wait} min</b> delay —
                    saving you around <b>{minutes_saved} min</b>.
                </div>

            </div>
            """
)

        else:

            st.info(
                f"🚧 {blocked_road} is reported blocked, but no "
                f"comparable alternative road could be scored right now."
            )

    st.markdown("")


    # --------------------------------------------------------
    # Input Summary
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Prediction Context</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    context_items = [
        (c1, "Direction", direction),
        (c2, "Road", road),
        (c3, "Time", selected_datetime.strftime("%H:%M")),
        (c4, "Temperature", f"{temp:.1f} °C"),
        (c5, "Rain", f"{rain:.1f} mm"),
        (c6, "Clouds", f"{clouds}%"),
    ]

    for col, label, value in context_items:
        with col:
            render_html(
                f"""
                <div class="context-card">
                    <div class="context-label">{label}</div>
                    <div class="context-value">{value}</div>
                </div>
                """
            )


# ============================================================
# SECTION: TRAFFIC ANALYTICS
# ============================================================

with st.expander("📊 Traffic Intelligence", expanded=False):

    # --------------------------------------------------------
    # Hourly Traffic
    # --------------------------------------------------------

    hourly = (
        df.groupby("hour")[
            "traffic_volume"
        ]
        .mean()
        .reset_index()
    )


    fig_hour = px.area(
        hourly,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic Volume by Hour",
        color_discrete_sequence=["#FF9F1C"]
    )

    fig_hour.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Hour",
        yaxis_title="Traffic Volume",
        hovermode="x unified",
        font_color="#1B2432"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Road & Weather Breakdown
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        # The dataset now stores the real district codes
        # (R3 District, R7 District) directly.
        road_traffic = (
            df.groupby("road_name")[
                "traffic_volume"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        fig_road = px.bar(
            road_traffic,
            x="traffic_volume",
            y="road_name",
            orientation="h",
            title="Top Roads by Average Traffic",
            color_discrete_sequence=["#06D6A0"]
        )

        fig_road.update_layout(
            yaxis_title="Road",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1B2432"
        )

        st.plotly_chart(
            fig_road,
            use_container_width=True
        )


    with col2:

        weather_traffic = (
            df.groupby("weather_main")[
                "traffic_volume"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig_weather = px.bar(
            weather_traffic,
            x="weather_main",
            y="traffic_volume",
            title="Traffic Volume by Weather",
            color_discrete_sequence=["#FFD166"]
        )

        fig_weather.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1B2432"
        )

        st.plotly_chart(
            fig_weather,
            use_container_width=True
        )


# ============================================================
# SECTION: MODEL PERFORMANCE
# ============================================================

with st.expander("🤖 Model Performance", expanded=False):

    if not model_results.empty:

        col1, col2 = st.columns(2)

        with col1:

            fig_rmse = px.bar(
                model_results,
                x="Model",
                y="RMSE",
                title="RMSE Comparison",
                text_auto=".2f",
                color_discrete_sequence=["#FF9F1C"]
            )

            fig_rmse.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1B2432"
            )

            st.plotly_chart(
                fig_rmse,
                use_container_width=True
            )

        with col2:

            fig_r2 = px.bar(
                model_results,
                x="Model",
                y="R2",
                title="R² Comparison",
                text_auto=".3f",
                color_discrete_sequence=["#06D6A0"]
            )

            fig_r2.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1B2432",
                yaxis=dict(
                    range=[
                        max(
                            0,
                            model_results["R2"].min() - 0.1
                        ),
                        1
                    ]
                )
            )

            st.plotly_chart(
                fig_r2,
                use_container_width=True
            )


# ============================================================
# SECTION: TRAFFIC TREND
# ============================================================

with st.expander("📈 Historical Traffic Trend", expanded=False):

    recent_data = df.tail(500).copy()


    fig_trend = go.Figure()

    fig_trend.add_trace(
        go.Scatter(
            x=recent_data["date_time"],
            y=recent_data["traffic_volume"],
            mode="lines",
            name="Traffic Volume",
            line=dict(
                width=2,
                color="#EF476F"
            )
        )
    )

    fig_trend.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time",
        yaxis_title="Traffic Volume",
        hovermode="x unified",
        font_color="#1B2432"
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Distribution & Rush-Hour Split
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig_dist = px.histogram(
            df,
            x="traffic_volume",
            nbins=50,
            title="Traffic Volume Distribution",
            color_discrete_sequence=["#118AB2"]
        )

        fig_dist.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1B2432"
        )

        st.plotly_chart(
            fig_dist,
            use_container_width=True
        )


    with col2:

        rush_data = (
            df.groupby(
                "is_rush_hour"
            )["traffic_volume"]
            .mean()
            .reset_index()
        )

        rush_data["is_rush_hour"] = (
            rush_data["is_rush_hour"]
            .map({
                0: "Normal Hours",
                1: "Rush Hours"
            })
        )

        fig_rush = px.pie(
            rush_data,
            names="is_rush_hour",
            values="traffic_volume",
            title="Traffic: Rush vs Normal Hours",
            color_discrete_sequence=["#06D6A0", "#FF9F1C"]
        )

        fig_rush.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1B2432"
        )

        st.plotly_chart(
            fig_rush,
            use_container_width=True
        )


# ============================================================
# SECTION: FOOTER
# ============================================================

render_html(
    """
    <div class="divider"></div>

    <div style="
        text-align:center;
        color:#64748B;
        padding:20px;
    ">

        <b>CapitalFlow AI</b>
        <br>
        Built to help the New Administrative Capital move a little smoother, one prediction at a time
        <br><br>
        Random Forest • Gradient Boosting • Neural Network

    </div>
    """
)
