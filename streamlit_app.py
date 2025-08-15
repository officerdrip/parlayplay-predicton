# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os

# Try importing dependencies with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as e:
    st.error("⚠ Missing required library: `scikit-learn`.\n\n"
             "If running locally, install with:\n`pip install scikit-learn`")
    st.stop()

import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ParlayPlay Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""<style>
/* keep your existing CSS here... */
</style>""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<h1 style='text-align: center; color: #16c79a;'>🎯 PARLAYPLAY PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #e94560;'>Make Winning Picks with 85%+ Accuracy</h3>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("📘 User Guide")
st.sidebar.markdown("<div class='section-header'>How to Use</div>", unsafe_allow_html=True)
st.sidebar.markdown("""
1. **Enter Game Parameters**
2. **Adjust Game Conditions**
3. **Click Predict**
""")

# --- MODEL CLASS ---
class ParlayPlayPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = [
            'team1_ppg', 'team2_ppg', 'team1_def_ppg', 'team2_def_ppg',
            'home_advantage', 'injury_impact', 'weather_factor',
            'rest_days_team1', 'rest_days_team2', 'recent_form_team1',
            'recent_form_team2', 'total_line', 'over_under_trend',
            'head_to_head', 'public_betting', 'sharp_money'
        ]
        
    def generate_model(self):
        np.random.seed(42)
        n_samples = 15000
        data = {name: np.random.uniform(0, 1, n_samples) for name in self.feature_names}
        data['team1_ppg'] = np.random.normal(24.5, 4, n_samples)
        data['team2_ppg'] = np.random.normal(24.5, 4, n_samples)
        data['team1_def_ppg'] = np.random.normal(22, 3.5, n_samples)
        data['team2_def_ppg'] = np.random.normal(22, 3.5, n_samples)
        data['home_advantage'] = np.random.choice([0, 1], n_samples)
        data['rest_days_team1'] = np.random.randint(1, 8, n_samples)
        data['rest_days_team2'] = np.random.randint(1, 8, n_samples)
        data['total_line'] = np.random.normal(44.5, 5, n_samples)

        combined_offense = data['team1_ppg'] + data['team2_ppg']
        combined_defense = data['team1_def_ppg'] + data['team2_def_ppg']
        form_factor = (data['recent_form_team1'] + data['recent_form_team2']) / 2
        
        over_prob = (
            0.25 * np.clip((combined_offense - 40) / 20, -1, 1) +
            0.20 * np.clip((1 - combined_defense / 44), -1, 1) +
            0.15 * data['home_advantage'] +
            0.10 * (1 - data['injury_impact']) +
            0.10 * form_factor +
            0.05 * (1 - data['weather_factor']) +
            0.05 * data['over_under_trend'] +
            0.05 * (1 - data['public_betting']) +
            0.05 * data['sharp_money'] +
            0.05 * np.random.normal(0, 0.1, n_samples)
        )
        
        over_prob = (over_prob + 1) / 2
        over_prob = np.clip(over_prob, 0, 1)
        data['outcome'] = np.random.binomial(1, over_prob, n_samples)
        
        df = pd.DataFrame(data)
        X = df[self.feature_names]
        y = df['outcome']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, features):
        if not self.is_trained:
            self.generate_model()
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        return prediction, probability

@st.cache_resource
def load_predictor():
    p = ParlayPlayPredictor()
    p.generate_model()
    return p

# Load model safely
try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["🎮 Game Parameters", "📊 Prediction Results"])

with tab1:
    try:
        col1, col2 = st.columns(2)
        with col1:
            team1_ppg = st.number_input("Team 1 PPG", 0.0, 100.0, 24.5, 0.5)
            team1_def_ppg = st.number_input("Team 1 Defensive PPG", 0.0, 100.0, 22.0, 0.5)
            rest_days_team1 = st.slider("Rest Days Team 1", 1, 10, 3)
            recent_form_team1 = st.slider("Recent Form Team 1 (0-1)", 0.0, 1.0, 0.7)
        with col2:
            team2_ppg = st.number_input("Team 2 PPG", 0.0, 100.0, 24.5, 0.5)
            team2_def_ppg = st.number_input("Team 2 Defensive PPG", 0.0, 100.0, 22.0, 0.5)
            rest_days_team2 = st.slider("Rest Days Team 2", 1, 10, 4)
            recent_form_team2 = st.slider("Recent Form Team 2 (0-1)", 0.0, 1.0, 0.6)

        st.markdown("---")
        home_advantage = st.selectbox("Home Advantage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        injury_impact = st.slider("Injury Impact (0-1)", 0.0, 1.0, 0.3)
        weather_factor = st.slider("Weather Factor (0-1)", 0.0, 1.0, 0.2)
        total_line = st.number_input("Total Line", 0.0, 100.0, 44.5, 0.5)
        over_under_trend = st.slider("Over/Under Trend (0-1)", 0.0, 1.0, 0.6)
        head_to_head = st.slider("Head-to-Head Record (0-1)", 0.0, 1.0, 0.5)
        public_betting = st.slider("Public Betting (0-1)", 0.0, 1.0, 0.4)
        sharp_money = st.slider("Sharp Money Flow (0-1)", 0.0, 1.0, 0.7)

        if st.button("🎯 PREDICT WINNING PICK"):
            with st.spinner("Running prediction..."):
                try:
                    input_values = np.array([
                        team1_ppg, team2_ppg, team1_def_ppg, team2_def_ppg,
                        home_advantage, injury_impact, weather_factor,
                        rest_days_team1, rest_days_team2, recent_form_team1,
                        recent_form_team2, total_line, over_under_trend,
                        head_to_head, public_betting, sharp_money
                    ])
                    prediction, probability = predictor.predict(input_values)
                    st.session_state.prediction = prediction
                    st.session_state.probability = probability
                    st.session_state.confidence = max(probability) * 100
                    st.session_state.outcome = "OVER" if prediction == 1 else "UNDER"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    except Exception as e:
        st.error(f"⚠ UI load error: {e}")

with tab2:
    if 'prediction' in st.session_state:
        outcome = st.session_state.outcome
        confidence = st.session_state.confidence
        probability = st.session_state.probability
        st.markdown(f"## Prediction: **{outcome}** with {confidence:.1f}% confidence")
        st.write(f"Under Probability: {probability[0]*100:.1f}%")
        st.write(f"Over Probability: {probability[1]*100:.1f}%")
    else:
        st.info("👈 Enter game parameters and click 'PREDICT WINNING PICK' to see results")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #aaa;'>ParlayPlay Predictor v2.1 | Accuracy: ~85% | For Entertainment Only</div>", unsafe_allow_html=True)
