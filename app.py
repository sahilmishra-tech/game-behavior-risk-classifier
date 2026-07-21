import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Gaming Behavior Risk Dashboard",
    page_icon="🎮",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
feature_names = list(model.feature_names_in_)

st.title("🎮 Gaming Behavior Risk Dashboard")
st.caption("Interactive predictions powered by your Random Forest classifier")

# ---- Sidebar inputs ----
st.sidebar.header("Player Profile")

age = st.sidebar.slider("Age", 13, 80, 25)
years_gaming = st.sidebar.slider("Years Gaming", 0.0, 50.0, 8.0, 0.5)
daily_playtime_hours = st.sidebar.slider("Daily Playtime (hours)", 0.0, 24.0, 4.0, 0.5)
weekly_play_sessions = st.sidebar.slider("Weekly Play Sessions", 0, 50, 5)
late_night_sessions_hours = st.sidebar.slider("Late-night Sessions (hours/week)", 0.0, 100.0, 8.0, 0.5)
weekend_playtime_hours = st.sidebar.slider("Weekend Playtime (hours)", 0.0, 48.0, 10.0, 0.5)
consecutive_hours_max = st.sidebar.slider("Longest Consecutive Session (hours)", 0.0, 48.0, 6.0, 0.5)

st.sidebar.header("Behavior & Wellbeing")
toxic_chat_reports = st.sidebar.number_input("Toxic Chat Reports", 0, 1000, 0)
rage_quit_frequency = st.sidebar.slider("Rage Quit Frequency", 0.0, 10.0, 2.0, 0.1)
in_game_purchases = st.sidebar.number_input("In-game Purchases", 0, 1000, 5)
monthly_spending_usd = st.sidebar.number_input("Monthly Spending (USD)", 0.0, 10000.0, 25.0, 5.0)
lootbox_openings = st.sidebar.number_input("Lootbox Openings", 0, 10000, 0)
stress_score = st.sidebar.slider("Stress Score", 0.0, 10.0, 5.0, 0.1)
loneliness_score = st.sidebar.slider("Loneliness Score", 0.0, 10.0, 5.0, 0.1)
dopamine_dependency_index = st.sidebar.slider("Dopamine Dependency Index", 0.0, 10.0, 5.0, 0.1)
self_control_score = st.sidebar.slider("Self-Control Score", 0.0, 10.0, 5.0, 0.1)
impulsiveness_score = st.sidebar.slider("Impulsiveness Score", 0.0, 10.0, 5.0, 0.1)
anxiety_level = st.sidebar.slider("Anxiety Level", 0.0, 10.0, 5.0, 0.1)
depression_indicator = st.sidebar.slider("Depression Indicator", 0.0, 10.0, 5.0, 0.1)
emotional_stability = st.sidebar.slider("Emotional Stability", 0.0, 10.0, 5.0, 0.1)

st.sidebar.header("Lifestyle & Performance")
sleep_hours = st.sidebar.slider("Sleep (hours/night)", 0.0, 16.0, 7.0, 0.5)
exercise_frequency_per_week = st.sidebar.slider("Exercise (sessions/week)", 0, 20, 3)
caffeine_intake_cups_day = st.sidebar.slider("Caffeine (cups/day)", 0.0, 20.0, 2.0, 0.5)
social_interaction_hours = st.sidebar.slider("Social Interaction (hours/week)", 0.0, 100.0, 10.0, 0.5)
gpa_or_performance_score = st.sidebar.slider("GPA / Performance Score", 0.0, 100.0, 75.0, 1.0)
missed_deadlines = st.sidebar.number_input("Missed Deadlines", 0, 1000, 0)
productivity_drop_percent = st.sidebar.slider("Productivity Drop (%)", 0.0, 100.0, 10.0, 1.0)
absenteeism_days = st.sidebar.number_input("Absenteeism Days", 0, 365, 0)
internet_speed_mbps = st.sidebar.number_input("Internet Speed (Mbps)", 0.0, 10000.0, 100.0, 5.0)
screen_time_total_hours = st.sidebar.slider("Total Screen Time (hours/day)", 0.0, 24.0, 8.0, 0.5)
burnout_probability = st.sidebar.slider("Burnout Probability", 0.0, 1.0, 0.25, 0.01)
mental_health_risk_score = st.sidebar.slider("Mental Health Risk Score", 0.0, 100.0, 25.0, 1.0)
churn_probability = st.sidebar.slider("Churn Probability", 0.0, 1.0, 0.25, 0.01)

st.sidebar.header("Categorical Profile")
gender = st.sidebar.selectbox("Gender", ["Male", "Non-binary", "Prefer not to say", "Other"])
country = st.sidebar.selectbox("Country", ["Brazil", "Canada", "China", "France", "Germany", "India", "Indonesia", "Japan", "Mexico", "Other", "Russia", "South Korea", "UK", "USA"])
occupation = st.sidebar.selectbox("Occupation", ["Freelancer", "Streamer/Content Creator", "Student", "Unemployed", "Other"])
income_level = st.sidebar.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "Other"])
preferred_genre = st.sidebar.selectbox("Preferred Genre", ["Casual", "FPS", "Horror", "MMORPG", "MOBA", "RPG", "Sandbox", "Sports", "Strategy", "Other"])
platform = st.sidebar.selectbox("Platform", ["Mobile", "PC", "PC+Console", "PC+Mobile", "Other"])
device_type = st.sidebar.selectbox("Device Type", ["High-end PC", "Laptop", "Mid-range PC", "Mixed", "Mobile", "Other"])
rank_tier = st.sidebar.selectbox("Rank Tier", ["Diamond", "Gold", "Grandmaster", "Master", "Platinum", "Silver", "Unranked", "Other"])
subscription_status = st.sidebar.selectbox("Subscription Status", ["None", "Premium", "Ultimate", "Other"])
relationship_status = st.sidebar.selectbox("Relationship Status", ["In a relationship", "Married", "Prefer not to say", "Single", "Other"])
behavioral_cluster = st.sidebar.selectbox("Behavioral Cluster", ["Casual Enjoyer", "Competitive Grinder", "Escape Seeker", "Social Gamer", "Streamer/Creator", "Toxic Competitor", "Other"])

def one_hot(row, prefix, value):
    key = f"{prefix}_{value}"
    if key in row:
        row[key] = 1

def make_input():
    row = {f: 0 for f in feature_names}
    numeric = {
        "age": age, "years_gaming": years_gaming,
        "daily_playtime_hours": daily_playtime_hours,
        "weekly_play_sessions": weekly_play_sessions,
        "late_night_sessions_hours": late_night_sessions_hours,
        "weekend_playtime_hours": weekend_playtime_hours,
        "consecutive_hours_max": consecutive_hours_max,
        "toxic_chat_reports": toxic_chat_reports,
        "rage_quit_frequency": rage_quit_frequency,
        "in_game_purchases": in_game_purchases,
        "monthly_spending_usd": monthly_spending_usd,
        "lootbox_openings": lootbox_openings,
        "stress_score": stress_score,
        "loneliness_score": loneliness_score,
        "dopamine_dependency_index": dopamine_dependency_index,
        "self_control_score": self_control_score,
        "impulsiveness_score": impulsiveness_score,
        "anxiety_level": anxiety_level,
        "depression_indicator": depression_indicator,
        "emotional_stability": emotional_stability,
        "sleep_hours": sleep_hours,
        "exercise_frequency_per_week": exercise_frequency_per_week,
        "caffeine_intake_cups_day": caffeine_intake_cups_day,
        "social_interaction_hours": social_interaction_hours,
        "gpa_or_performance_score": gpa_or_performance_score,
        "missed_deadlines": missed_deadlines,
        "productivity_drop_percent": productivity_drop_percent,
        "absenteeism_days": absenteeism_days,
        "internet_speed_mbps": internet_speed_mbps,
        "screen_time_total_hours": screen_time_total_hours,
        "burnout_probability": burnout_probability,
        "mental_health_risk_score": mental_health_risk_score,
        "churn_probability": churn_probability,
        "multiplayer_ratio": 0.5,
    }
    row.update({k: v for k, v in numeric.items() if k in row})
    for prefix, value in {
        "gender": gender,
        "country": country,
        "occupation": occupation,
        "income_level": income_level,
        "preferred_genre": preferred_genre,
        "platform": platform,
        "device_type": device_type,
        "rank_tier": rank_tier,
        "subscription_status": subscription_status,
        "relationship_status": relationship_status,
        "behavioral_cluster": behavioral_cluster,
    }.items():
        one_hot(row, prefix, value)
    return pd.DataFrame([row], columns=feature_names)

X = make_input()
prediction = int(model.predict(X)[0])
probabilities = model.predict_proba(X)[0]
risk_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

# ---- Main dashboard ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
col2.metric("Risk Probability", f"{risk_probability:.1%}")
col3.metric("Model", "Random Forest")
col4.metric("Trees", getattr(model, "n_estimators", "N/A"))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Risk Probability")
    chart_df = pd.DataFrame({
        "Class": ["Low Risk", "High Risk"],
        "Probability": probabilities
    }).set_index("Class")
    st.bar_chart(chart_df)

with right:
    st.subheader("Player Snapshot")
    snapshot = pd.DataFrame({
        "Metric": [
            "Daily playtime", "Sleep", "Stress", "Self-control",
            "Mental health risk", "Productivity drop"
        ],
        "Value": [
            f"{daily_playtime_hours:.1f} hours",
            f"{sleep_hours:.1f} hours",
            f"{stress_score:.1f}/10",
            f"{self_control_score:.1f}/10",
            f"{mental_health_risk_score:.0f}/100",
            f"{productivity_drop_percent:.0f}%"
        ]
    })
    st.dataframe(snapshot, hide_index=True, use_container_width=True)

st.subheader("Model Input Data")
st.dataframe(X.T.rename(columns={0: "Value"}), use_container_width=True)

st.info(
    "This dashboard is a machine-learning prediction interface. "
    "The prediction is not a medical or clinical diagnosis."
)
