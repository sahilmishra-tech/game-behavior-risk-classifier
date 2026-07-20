import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Gaming Behavioral & Risk Classification",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Gaming Behavioral & Risk Analysis App")
st.markdown("""
This application uses a pre-trained **Random Forest Classifier** to analyze gaming habits, 
psychological indicators, lifestyle factors, and demographics to predict user classifications.
""")

# Load the Model
@st.cache_resource
def load_model():
    try:
        # Load the model pickle file saved in your directory
        model = joblib.load("random_forest_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

model = load_model()

if model is not None:
    # Extract Feature Names from Model Metadata
    feature_names = list(model.feature_names_in_)

    st.sidebar.header("📋 Input Feature Parameters")
    
    # Define Numerical Features default ranges
    num_defaults = {
        "age": (18, 10, 80),
        "years_gaming": (5, 0, 50),
        "daily_playtime_hours": (3.0, 0.0, 24.0),
        "weekly_play_sessions": (10, 0, 100),
        "late_night_sessions_hours": (2.0, 0.0, 12.0),
        "weekend_playtime_hours": (8.0, 0.0, 48.0),
        "consecutive_hours_max": (4.0, 0.0, 24.0),
        "multiplayer_ratio": (0.5, 0.0, 1.0),
        "toxic_chat_reports": (0, 0, 50),
        "rage_quit_frequency": (1, 0, 20),
        "in_game_purchases": (1, 0, 100),
        "monthly_spending_usd": (20.0, 0.0, 5000.0),
        "lootbox_openings": (0, 0, 100),
        "stress_score": (5.0, 0.0, 10.0),
        "loneliness_score": (5.0, 0.0, 10.0),
        "dopamine_dependency_index": (5.0, 0.0, 10.0),
        "self_control_score": (5.0, 0.0, 10.0),
        "impulsiveness_score": (5.0, 0.0, 10.0),
        "anxiety_level": (5.0, 0.0, 10.0),
        "depression_indicator": (0.0, 0.0, 1.0),
        "emotional_stability": (5.0, 0.0, 10.0),
        "sleep_hours": (7.0, 0.0, 24.0),
        "exercise_frequency_per_week": (3, 0, 14),
        "caffeine_intake_cups_day": (2, 0, 10),
        "social_interaction_hours": (2.0, 0.0, 24.0),
        "gpa_or_performance_score": (3.0, 0.0, 4.0),
        "missed_deadlines": (1, 0, 20),
        "productivity_drop_percent": (10.0, 0.0, 100.0),
        "absenteeism_days": (1, 0, 30),
        "internet_speed_mbps": (100.0, 1.0, 2000.0),
        "screen_time_total_hours": (8.0, 0.0, 24.0),
        "burnout_probability": (0.2, 0.0, 1.0),
        "mental_health_risk_score": (0.3, 0.0, 1.0),
        "churn_probability": (0.1, 0.0, 1.0),
    }

    input_data = {}

    st.sidebar.subheader("Numerical Features")
    for feat in num_defaults:
        if feat in feature_names:
            default, min_val, max_val = num_defaults[feat]
            if isinstance(default, float):
                input_data[feat] = st.sidebar.slider(feat, float(min_val), float(max_val), float(default))
            else:
                input_data[feat] = st.sidebar.slider(feat, int(min_val), int(max_val), int(default))

    st.sidebar.subheader("Categorical Feature Groups")
    
    # Helper to handle One-Hot Encoded features
    def handle_categorical(group_prefix, options, default_idx=0):
        selected = st.sidebar.selectbox(group_prefix, options, index=default_idx)
        for opt in options:
            col_name = f"{group_prefix}_{opt}"
            if col_name in feature_names:
                input_data[col_name] = 1.0 if opt == selected else 0.0

    handle_categorical("gender", ["Male", "Non-binary", "Prefer not to say", "Female"])
    handle_categorical("country", ["Brazil", "Canada", "China", "France", "Germany", "India", "Indonesia", "Japan", "Mexico", "Other", "Russia", "South Korea", "UK", "USA"])
    handle_categorical("occupation", ["Freelancer", "Streamer/Content Creator", "Student", "Unemployed", "Other"])
    handle_categorical("income_level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
    handle_categorical("preferred_genre", ["Casual", "FPS", "Horror", "MMORPG", "MOBA", "RPG", "Sandbox", "Sports", "Strategy"])
    handle_categorical("platform", ["Mobile", "PC", "PC+Console", "PC+Mobile", "Console"])
    handle_categorical("device_type", ["High-end PC", "Laptop", "Mid-range PC", "Mixed", "Mobile"])
    handle_categorical("rank_tier", ["Diamond", "Gold", "Grandmaster", "Master", "Platinum", "Silver", "Unranked"])
    handle_categorical("subscription_status", ["None", "Premium", "Ultimate"])
    handle_categorical("relationship_status", ["In a relationship", "Married", "Prefer not to say", "Single"])
    handle_categorical("behavioral_cluster", ["Casual Enjoyer", "Competitive Grinder", "Escape Seeker", "Social Gamer", "Streamer/Creator", "Toxic Competitor"])

    # Ensure all features from model are populated in correct sequence
    df_input = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0.0
            
    df_input = df_input[feature_names]

    # Display Input Summary
    st.subheader("Input Profile Summary")
    st.dataframe(df_input.T.rename(columns={0: "Value"}), height=300)

    # Predict
    if st.button("🚀 Predict Classification", type="primary"):
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]

        st.success(f"**Predicted Class:** `{prediction}`")
        
        st.subheader("Class Probabilities")
        proba_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": prediction_proba
        })
        st.bar_chart(proba_df.set_index("Class"))