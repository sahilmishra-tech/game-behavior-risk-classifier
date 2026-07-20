https://game-behavior-risk-classifier-3re345bdemhmtpgevvntnx.streamlit.app/

# 🎮 Gaming Behavioral & Risk Classifier

An interactive Machine Learning web application built using **Streamlit** and **Scikit-Learn**. This app leverages a trained **Random Forest Classifier** to analyze player demographics, psychological indicators, lifestyle habits, and in-game behavior to predict player classifications and risk profiles.

---

## 🚀 Key Features

* **Interactive Sidebar Controls:** Adjust numerical sliders and categorical select boxes in real-time.
* **Automated Data Preprocessing:** Encodes user inputs directly into the feature format expected by the model.
* **Instant Predictions:** Generates target classifications along with full probability distribution charts.
* **Input Feature Summary:** Displays a clean preview table of configured inputs before running predictions.

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application script
├── random_forest_model.pkl     # Pre-trained Random Forest model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
