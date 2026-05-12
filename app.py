"""
Student Dropout Risk Predictor — Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json, os

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
)

# ─── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    art  = os.path.join(base, "artifacts")
    with open(os.path.join(art, "rf_model.pkl"),       "rb") as f: model    = pickle.load(f)
    with open(os.path.join(art, "scaler.pkl"),          "rb") as f: scaler   = pickle.load(f)
    with open(os.path.join(art, "imputer.pkl"),         "rb") as f: imputer  = pickle.load(f)
    with open(os.path.join(art, "feature_names.json"),  "r")  as f: features = json.load(f)
    with open(os.path.join(art, "top_features.json"),   "r")  as f: top5     = json.load(f)
    with open(os.path.join(art, "metrics.json"),        "r")  as f: metrics  = json.load(f)
    return model, scaler, imputer, features, top5, metrics

model, scaler, imputer, features, top5, metrics = load_artifacts()

# ─── Header ────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#2c3e50;'>🎓 Student Dropout Risk Predictor</h1>
<p style='text-align:center; color:#7f8c8d; font-size:16px;'>
    Enter student details below to estimate dropout risk and identify key contributing factors.
</p>
<hr style='border:1px solid #ecf0f1;'>
""", unsafe_allow_html=True)

# ─── Sidebar — model performance ───────────────────────────────
with st.sidebar:
    st.markdown("### Model Performance")
    em = metrics["eval_metrics"]
    for mname in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        m = em[mname]
        selected = "⭐ " if mname == "Random Forest" else ""
        st.markdown(f"**{selected}{mname}**")
        col1, col2 = st.columns(2)
        col1.metric("AUC",  f"{m['auc_roc']:.3f}")
        col2.metric("F1",   f"{m['f1']:.3f}")
    st.markdown("---")
    st.caption("Model: Random Forest (150 trees, SMOTE balanced)\nDataset: UCI Student Dropout (n=4,424)")

# ─── Input form ────────────────────────────────────────────────
st.markdown("## Student Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📋 Academic Background**")
    age            = st.slider("Age at Enrollment",        17, 60, 20)
    prev_grade     = st.slider("Previous Qualification Grade", 60.0, 200.0, 130.0, step=0.5)
    adm_grade      = st.slider("Admission Grade",          60.0, 200.0, 135.0, step=0.5)
    prev_qual      = st.selectbox("Previous Qualification Type", [1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: {1:"Secondary", 2:"Higher Ed.", 3:"Degree",
                                                           4:"Masters", 5:"Doctorate", 6:"Other"}[x])

with col2:
    st.markdown("**📚 Curricular Performance**")
    units_enr      = st.slider("Curricular Units Enrolled",  0, 12, 6)
    units_appr     = st.slider("Curricular Units Approved",  0, 12, 5)
    units_grade    = st.slider("Curricular Units Grade (avg)", 0.0, 20.0, 13.0, step=0.1)

with col3:
    st.markdown("**💼 Socioeconomic Factors**")
    debtor         = st.selectbox("Has Outstanding Debt?",        ["No", "Yes"])
    tuition_ok     = st.selectbox("Tuition Fees Up to Date?",     ["Yes", "No"])
    scholarship    = st.selectbox("Scholarship Holder?",          ["No", "Yes"])
    gender         = st.selectbox("Gender",                       ["Female", "Male"])
    marital        = st.selectbox("Marital Status",               [1,2,3,4,5,6],
                                   format_func=lambda x: {1:"Single",2:"Married",3:"Widower",
                                                           4:"Divorced",5:"Civil Union",6:"Separated"}[x])
    nationality    = st.slider("Nationality Code (1–21)",         1, 21, 1)

# ─── Predict button ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Predict Dropout Risk", use_container_width=True, type="primary")

if predict_btn:
    raw = np.array([[
        age, prev_grade, adm_grade,
        units_enr, units_appr, units_grade,
        1 if debtor     == "Yes" else 0,
        1 if tuition_ok == "Yes" else 0,
        1 if scholarship== "Yes" else 0,
        1 if gender     == "Male" else 0,
        marital, nationality, prev_qual,
    ]])

    X_imp    = imputer.transform(raw)
    X_scaled = scaler.transform(X_imp)
    prob     = model.predict_proba(X_scaled)[0][1]
    risk_pct = prob * 100

    # Risk label
    if risk_pct < 30:
        level, color, emoji = "Low",    "#27ae60", "🟢"
    elif risk_pct < 60:
        level, color, emoji = "Medium", "#f39c12", "🟡"
    else:
        level, color, emoji = "High",   "#e74c3c", "🔴"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown(f"""
        <div style='background:{color}22; border:2px solid {color}; border-radius:12px;
                    padding:28px; text-align:center;'>
            <div style='font-size:48px;'>{emoji}</div>
            <div style='font-size:36px; font-weight:bold; color:{color};'>{risk_pct:.1f}%</div>
            <div style='font-size:22px; color:{color}; font-weight:600;'>{level} Risk</div>
            <div style='font-size:13px; color:#7f8c8d; margin-top:8px;'>Dropout Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### Top Contributing Risk Factors")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        bar_colors = [color if i < 3 else "#bdc3c7" for i in range(len(feat_imp))]
        ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=bar_colors[::-1], edgecolor="white")
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title("Feature Importance (Random Forest)", fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Gauge-style progress bar
    st.markdown("#### Risk Meter")
    st.progress(int(risk_pct))
    st.caption(f"Risk Score: {risk_pct:.1f} / 100")

    # Top factor explanation
    st.markdown("#### What contributes most to this student's risk?")
    factor_labels = {
        "Curricular_units_approved":        ("Units Approved", "Few approved units signal academic difficulty."),
        "Curricular_units_grade":           ("Unit Grades",    "Low average grades increase dropout likelihood."),
        "Curricular_units_enrolled":        ("Units Enrolled", "Enrolling in fewer units may indicate disengagement."),
        "Previous_qualification_grade":     ("Prior Grade",    "Lower prior academic performance predicts risk."),
        "Age_at_enrollment":                ("Age",            "Older enrollment age is associated with higher dropout rates."),
        "Admission_grade":                  ("Admission Grade","Students with lower admission scores face more challenges."),
        "Debtor":                           ("Debt Status",    "Having outstanding debts is a strong dropout indicator."),
        "Tuition_fees_up_to_date":          ("Tuition Status", "Unpaid tuition is a major financial stress signal."),
        "Scholarship_holder":               ("Scholarship",    "Scholarship holders tend to have more retention incentive."),
    }
    for i, feat in enumerate(top5[:3], 1):
        label, explanation = factor_labels.get(feat, (feat, "This feature contributes significantly to risk."))
        st.markdown(f"**{i}. {label}** — {explanation}")

    # Disclaimer
    st.info(
        "⚠️ **Disclaimer:** This tool is a decision-support aid only. Predictions are based on statistical "
        "patterns and should not replace professional academic counselling. Individual circumstances may "
        "differ significantly from model assumptions."
    )
