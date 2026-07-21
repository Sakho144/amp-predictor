#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from modlamp.descriptors import GlobalDescriptor
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="AMP-Predictor", page_icon="🧬", layout="wide")

# ---------- CHARGEMENT MODÈLE ET SCALER ----------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_xgb_final_100.pkl")
        scaler = joblib.load("scaler_final_100.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Model files not found: {e}")
        st.stop()

model, scaler = load_model()

# ---------- SHAP EXPLAINER ----------
@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

shap_explainer = load_shap_explainer()

# ---------- CHARGEMENT DU DATASET DE RÉFÉRENCE ----------
@st.cache_data
def load_reference_data():
    df = pd.read_csv("peptide_features_length_balanced.csv")
    sequences_ref = df["sequence"].tolist()
    labels_ref = df["activity_label_final"].tolist()
    return sequences_ref, labels_ref

seq_ref, labels_ref = load_reference_data()

# ---------- CONSTANTES ----------
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
FEATURE_NAMES = [
    'global_0', 'global_1', 'global_2', 'global_3', 'global_4',
    'global_5', 'global_6', 'global_7', 'global_8', 'global_9',
    'aa_A', 'aa_C', 'aa_D', 'aa_E', 'aa_F', 'aa_G', 'aa_H', 'aa_I', 'aa_K',
    'aa_L', 'aa_M', 'aa_N', 'aa_P', 'aa_Q', 'aa_R', 'aa_S', 'aa_T', 'aa_V',
    'aa_W', 'aa_Y', 'length', 'net_charge', 'hydrophobicity'
]

# ---------- FONCTIONS ----------
def compute_features(sequence):
    sequence = sequence.upper().strip()
    if len(sequence) < 5 or len(sequence) > 50:
        return None, f"Length must be 5-50 AA (got {len(sequence)})."
    invalid = set(sequence) - VALID_AA
    if invalid:
        return None, f"Invalid amino acids: {', '.join(sorted(invalid))}. Use only ACDEFGHIKLMNPQRSTVWY."
    try:
        gd = GlobalDescriptor([sequence])
        gd.calculate_all()
        global_desc = gd.descriptor[0]
        if isinstance(global_desc, np.ndarray):
            global_desc = global_desc.tolist()
        if len(global_desc) < 10:
            global_desc = global_desc + [0.0] * (10 - len(global_desc))
        else:
            global_desc = global_desc[:10]
    except Exception as e:
        return None, f"Descriptor calculation error: {e}"
    aas = "ACDEFGHIKLMNPQRSTVWY"
    length = len(sequence)
    aa_comp = [sequence.count(aa) / length for aa in aas]
    net_charge = sequence.count('K') + sequence.count('R') - sequence.count('D') - sequence.count('E')
    hydro_scale = {
        'I':4.5, 'V':4.2, 'L':3.8, 'F':2.8, 'C':2.5, 'M':1.9, 'A':1.8,
        'G':-0.4, 'T':-0.7, 'S':-0.8, 'W':-0.9, 'Y':-1.3, 'P':-1.6, 'H':-3.2,
        'E':-3.5, 'Q':-3.5, 'D':-3.5, 'N':-3.5, 'K':-3.9, 'R':-4.5
    }
    hydrophobicity = sum(hydro_scale.get(aa, 0) for aa in sequence) / length
    features = global_desc + aa_comp + [length, net_charge, hydrophobicity]
    return np.array(features, dtype=np.float64).reshape(1, -1), None

def get_properties(features):
    f = features.flatten()
    props = {
        "Length (AA)": int(f[FEATURE_NAMES.index('length')]),
        "MW (Da)": round(f[FEATURE_NAMES.index('global_1')], 2),
        "Net charge": round(f[FEATURE_NAMES.index('net_charge')], 2),
        "pI": round(f[FEATURE_NAMES.index('global_4')], 2),
        "Hydrophobicity (KD)": round(f[FEATURE_NAMES.index('hydrophobicity')], 4)
    }
    return props

def display_shap_local(features_scaled):
    shap_values = shap_explainer.shap_values(features_scaled)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0] if len(shap_values) == 2 else shap_values[0][0]
        expected = shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, list) else shap_explainer.expected_value
    else:
        if shap_values.ndim == 2:
            shap_vals = shap_values[0]
        elif shap_values.ndim == 3:
            shap_vals = shap_values[0, :, 1]
        else:
            st.error("Unexpected SHAP shape")
            return
        expected = shap_explainer.expected_value
        if isinstance(expected, list):
            expected = expected[1] if len(expected) > 1 else expected[0]
    exp = shap.Explanation(values=shap_vals, base_values=expected, data=features_scaled[0], feature_names=FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(exp, max_display=15)
    plt.title("Local SHAP explanation - Descriptor contributions")
    st.pyplot(fig)
    plt.close()

def align_sequences(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    seq1_aligned = seq1[:min_len]
    seq2_aligned = seq2[:min_len]
    matches = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a == b)
    identity = matches / min_len if min_len > 0 else 0
    html = '<div style="font-family: monospace; font-size: 14px; line-height: 1.4;">'
    html += '<div style="margin-bottom: 4px;">'
    for a, b in zip(seq1_aligned, seq2_aligned):
        if a == b:
            html += f'<span style="background-color: #c8e6c9; padding: 0 2px;">{a}</span>'
        else:
            html += f'<span style="background-color: #ffcdd2; color: #c62828; padding: 0 2px;">{a}</span>'
    html += '</div><div style="margin-bottom: 4px;">'
    for a, b in zip(seq1_aligned, seq2_aligned):
        html += '|' if a == b else ' '
    html += '</div><div>'
    for a, b in zip(seq1_aligned, seq2_aligned):
        if a == b:
            html += f'<span style="background-color: #c8e6c9; padding: 0 2px;">{b}</span>'
        else:
            html += f'<span style="background-color: #ffcdd2; color: #c62828; padding: 0 2px;">{b}</span>'
    html += '</div></div>'
    return html, identity

def find_most_similar_by_identity(input_seq, ref_seqs, ref_labels):
    best_ratio = 0
    best_seq = None
    best_label = None
    for seq, label in zip(ref_seqs, ref_labels):
        ratio = SequenceMatcher(None, input_seq, seq).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_seq = seq
            best_label = "Active" if label == 1 else "Inactive"
    return best_seq, best_label, best_ratio

def get_confidence_level(identity_ratio):
    if identity_ratio > 0.8:
        return "Very high", "🟢"
    elif identity_ratio > 0.5:
        return "High", "🟡"
    elif identity_ratio > 0.3:
        return "Moderate", "🟠"
    else:
        return "Low", "🔴"

# ---------- INTERFACE ----------
st.sidebar.title("🧬 AMP-Predictor")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🧪 Prediction", "📊 Performance"])

# ---------- PAGE HOME ----------
if page == "🏠 Home":
    st.title("AMP-Predictor: Antimicrobial Peptide Activity Prediction")
    st.markdown("---")

    with st.expander("📖 About this tool", expanded=True):
        st.markdown("""
        **AMP-Predictor** is a machine learning web application that predicts whether a given peptide sequence has antimicrobial activity.  
        It is designed for researchers in microbiology, bioinformatics, and drug discovery who need to rapidly screen potential antimicrobial peptides (AMPs).
        """)

    with st.expander("⚙️ How it works", expanded=False):
        st.markdown("""
        - **Input**: Peptide sequence (5–50 amino acids, uppercase letters A–Y).  
        - **Descriptors**: 33 features including amino acid composition, length, net charge, hydrophobicity, and 10 global descriptors (MW, pI, aliphatic index, Boman index, etc.) computed with `modlAMP`.  
        - **Model**: XGBoost classifier trained on a length-bias-corrected, quality-filtered dataset of 2,830 peptides (1,415 active, 1,415 inactive).  
        - **Output**: Probability of being active (0–100%) and a binary prediction (Active/Inactive), plus a local SHAP explanation and similarity search.
        """)

    # Key results (blind test, 90/10 split model — reported in the article)
    st.subheader("📊 Model performance (blind test)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC", "0.802")
    col2.metric("F1-score", "0.719")
    col3.metric("Accuracy", "0.721")
    col4.metric("Precision", "0.721")
    col5.metric("Recall", "0.716")
    st.markdown("Blind test on 10% of the data (283 peptides never used during training). The model deployed in this app is retrained on 100% of the dataset for maximal data usage; the metrics above come from the held-out evaluation reported in the article.")

    with st.expander("⚠️ Limitations", expanded=False):
        st.markdown("""
        - **Short peptides (<10 AA)**: Predictions are less reliable due to underrepresentation in the training set.  
        - **General activity only**: The model does not distinguish between Gram-positive, Gram-negative, or fungal targets.  
        - **No MIC prediction**: The model outputs only the probability of activity, not the minimal inhibitory concentration.  
        - **Sequence length**: Only peptides between 5 and 50 amino acids, using standard L-amino acids, are accepted.  
        - **Experimental validation**: Predictions should be interpreted with caution and validated experimentally.
        """)

    with st.expander("🤝 Collaboration & Contact", expanded=False):
        st.markdown("""
        If you are interested in collaborating, testing the tool on your datasets, or need assistance, please contact:  
        **Email**: `mahamadou.sakho@etu.uae.ac.ma`  
        (Feel free to contact us)
        """)

# ---------- PAGE PREDICTION ----------
elif page == "🧪 Prediction":
    st.title("Predict antimicrobial activity")
    st.markdown("Enter a peptide sequence (5-50 AA, uppercase).")

    if 'seq_input' not in st.session_state:
        st.session_state.seq_input = ""

    sequence = st.text_area("Peptide sequence", height=100, value=st.session_state.seq_input)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load active example (Magainin-2)"):
            st.session_state.seq_input = "GIGKFLHSAKKFGKAFVGEIMNS"
            st.rerun()
    with col2:
        if st.button("Load inactive example (Poly-Alanine)"):
            st.session_state.seq_input = "AAAAAAAAAAAAAAAAAAAA"
            st.rerun()

    if st.button("Predict", type="primary"):
        if not sequence:
            st.warning("Please enter a sequence or use an example.")
        else:
            features, err = compute_features(sequence)
            if err:
                st.error(err)
            else:
                features_scaled = scaler.transform(features)
                proba = float(model.predict_proba(features_scaled)[0, 1])
                prediction = "Active" if proba >= 0.5 else "Inactive"
                st.success(f"### Prediction: **{prediction}** (probability {proba:.2%})")
                st.progress(proba)

                if len(sequence) <= 10:
                    st.warning("⚠️ Short peptide (<10 AA) – prediction reliability may be lower.")

                # Similarity and confidence
                best_seq, best_label, best_ratio = find_most_similar_by_identity(sequence, seq_ref, labels_ref)
                st.markdown("---")
                st.subheader("🔍 Similarity with known peptides")
                conf_level, conf_icon = get_confidence_level(best_ratio)
                st.info(f"{conf_icon} **Confidence level:** {conf_level} (based on sequence identity with training set)")

                if best_ratio > 0.95:
                    st.success(f"**Identical to known peptide** (sequence identity = {best_ratio:.1%})")
                    st.markdown(f"Known peptide: `{best_seq}` ({best_label})")
                    align_html, _ = align_sequences(sequence, best_seq)
                    st.markdown(align_html, unsafe_allow_html=True)
                elif best_ratio > 0.7:
                    st.info(f"**Similar to known peptide** (sequence identity = {best_ratio:.1%})")
                    st.markdown(f"Known peptide: `{best_seq}` ({best_label})")
                    align_html, _ = align_sequences(sequence, best_seq)
                    st.markdown(align_html, unsafe_allow_html=True)
                else:
                    st.info(f"No highly similar peptide found (best identity = {best_ratio:.1%}). Prediction relies on general patterns.")

                # Properties
                st.subheader("Calculated properties")
                props = get_properties(features)
                st.dataframe(pd.DataFrame(props.items(), columns=["Property", "Value"]), hide_index=True)

                # SHAP
                st.subheader("SHAP explanation")
                try:
                    display_shap_local(features_scaled)
                except Exception as e:
                    st.warning(f"SHAP plot not available: {e}")

# ---------- PAGE PERFORMANCE ----------
elif page == "📊 Performance":
    st.title("Model performance (general model)")

    st.markdown("### Blind test results")
    metrics = {"AUC": 0.8022, "F1-score": 0.7189, "Accuracy": 0.7208, "Precision": 0.7214, "Recall": 0.7163, "MCC": 0.4417}
    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
        - **AUC (Area Under the ROC Curve)**: Measures the model's ability to distinguish active from inactive peptides.
        - **F1-score**: Harmonic mean of precision and recall. Balances false positives and false negatives.
        - **Accuracy**: Overall proportion of correct predictions.
        - **Precision**: Among peptides predicted as active, how many are truly active.
        - **Recall**: Among truly active peptides, how many were correctly identified.
        - **MCC (Matthews Correlation Coefficient)**: Balanced measure robust to class imbalance, ranges from -1 to +1.
        """)

    st.markdown("---")
    st.markdown("### Training dataset characteristics")
    st.markdown("""
    - **Total peptides**: 2,830 (1,415 active, 1,415 inactive), length-bias corrected  
    - **Length range**: 5–50 amino acids (mean ≈ 17.2 AA for active, ≈ 15.9 AA for inactive)  
    - **Descriptors**: 33 features (10 global, 20 AA composition, length, net charge, hydrophobicity)  
    - **Model**: XGBoost (n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_lambda=1, reg_alpha=0)  
    - **Deployment note**: the model used in this app is retrained on 100% of the dataset above; metrics reported here come from a held-out 10% blind test evaluation of an equivalent model (90/10 split), for an honest estimate of generalization performance.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("XGBoost – Blind AUC = 0.802")
