# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import shap

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Cybersecurity SOC Dashboard",
    layout="wide"
)

st.title("🛡️ AI & ML Cybersecurity – Predicting & Preventing Future Threats")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("cyber_threat_dataset.csv")

df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("🔍 Filters")

attack_filter = st.sidebar.multiselect(
    "Select Attack Types",
    df["attack_type"].unique(),
    default=df["attack_type"].unique()
)

df_filtered = df[df["attack_type"].isin(attack_filter)]

# =========================
# DATA PREVIEW
# =========================
st.subheader("📘 Dataset Preview")
st.dataframe(df_filtered, use_container_width=True)

# =========================
# VISUAL ANALYTICS
# =========================
st.subheader("📊 Threat Analytics")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        px.histogram(df_filtered, x="attack_type", title="Attack Distribution"),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        px.scatter(
            df_filtered,
            x="packet_size",
            y="cpu_usage",
            color="attack_type",
            title="Packet Size vs CPU Usage"
        ),
        use_container_width=True
    )

# =========================
# FEATURE SET
# =========================
features = [
    "packet_size",
    "port",
    "failed_logins",
    "login_frequency",
    "cpu_usage",
    "api_calls",
    "file_entropy",
    "anomaly_score"
]

X = df[features]
y = df["attack_type"]

encoder = LabelEncoder()
y_encoded = pd.Series(encoder.fit_transform(y))

# =========================
# TRAIN TEST SPLIT (SAFE)
# =========================
min_count = y_encoded.value_counts().min()

if min_count < 2:
    st.warning("⚠️ Small class samples detected – stratification disabled")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

# =========================
# RANDOM FOREST MODEL
# =========================
st.subheader("🤖 Random Forest – Threat Classification")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

labels = list(range(len(encoder.classes_)))

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=encoder.classes_,
    zero_division=0
)

st.text("📄 Classification Report")
st.text(report)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

st.subheader("📉 Confusion Matrix")
st.dataframe(cm_df, use_container_width=True)

# =========================
# ROC–AUC
# =========================
st.subheader("📈 ROC–AUC Analysis")

y_prob = rf_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
st.metric("Overall ROC–AUC Score", round(roc_auc, 3))

malware_index = list(encoder.classes_).index("Malware")

fpr, tpr, _ = roc_curve(
    (y_test == malware_index).astype(int),
    y_prob[:, malware_index]
)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Malware Detection")
st.pyplot(plt)

# =========================
# SHAP EXPLAINABILITY
# =========================
st.subheader("🔍 Explainable AI (SHAP)")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=features,
    show=False
)
st.pyplot(bbox_inches="tight")

# =========================
# ZERO-DAY DETECTION (AUTOENCODER)
# =========================
st.subheader("🧠 Zero-Day Detection – Autoencoder")

normal_data = df[df["attack_type"] == "Normal"][features]

scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_data)

input_dim = normal_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(
    normal_scaled,
    normal_scaled,
    epochs=10,
    batch_size=64,
    verbose=0
)

test_scaled = scaler.transform(df[features])
reconstructed = autoencoder.predict(test_scaled, verbose=0)
recon_error = np.mean(np.square(test_scaled - reconstructed), axis=1)

df["reconstruction_error"] = recon_error
threshold = np.percentile(recon_error, 95)
df["ZeroDay_Predicted"] = df["reconstruction_error"] > threshold

st.metric("Detected Zero-Day Events", int(df["ZeroDay_Predicted"].sum()))

# =========================
# LSTM ATTACK FORECASTING
# =========================
st.subheader("⏳ LSTM – Attack Forecasting")

df["attack_label"] = encoder.transform(df["attack_type"])

time_features = [
    "packet_size",
    "cpu_usage",
    "login_frequency",
    "file_entropy",
    "attack_label"
]

scaler_lstm = MinMaxScaler()
scaled_data = scaler_lstm.fit_transform(df[time_features])

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][-1])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_data)

split = int(0.8 * len(X_lstm))
X_lstm_train, X_lstm_test = X_lstm[:split], X_lstm[split:]
y_lstm_train, y_lstm_test = y_lstm[:split], y_lstm[split:]

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])),
    LSTM(32),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")

lstm_model.fit(
    X_lstm_train,
    y_lstm_train,
    epochs=5,
    batch_size=64,
    verbose=0
)

pred_future = lstm_model.predict(X_lstm_test, verbose=0)

st.success("✅ LSTM trained – future attack behavior can be forecasted")

# =========================
# REAL-TIME PREDICTION
# =========================
st.subheader("🚨 Real-Time Threat Prediction")

user_input = []
for f in features:
    user_input.append(st.number_input(f, value=float(df[f].mean())))

if st.button("Predict Threat"):
    pred = rf_model.predict([user_input])[0]
    attack = encoder.inverse_transform([pred])[0]

    st.error(f"⚠️ Predicted Threat: **{attack}**")

    if attack in ["DoS", "DDoS"]:
        st.warning("🛑 Block IP & Apply Rate Limiting")
    elif attack == "Malware":
        st.warning("🧪 Isolate Host & Run Malware Scan")
    elif attack == "Phishing":
        st.warning("📧 Block URL & Alert Users")
    elif attack == "BruteForce":
        st.warning("🔐 Lock Account")
    elif attack == "InsiderThreat":
        st.warning("👤 Revoke Privileges")
    elif attack == "ZeroDay":
        st.warning("🧠 Quarantine & Manual Analysis")
    else:
        st.success("✅ Normal Activity")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🔐 **AI-Powered Cybersecurity SOC Dashboard**")
