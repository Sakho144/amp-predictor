#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from modlamp.descriptors import GlobalDescriptor
from difflib import SequenceMatcher
from Bio import Align
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="AMP-Predictor", page_icon="🧬", layout="wide")

# ---------- CHARGEMENT MODÈLE ET SCALER ----------
@st.cache_resource
def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model("model_xgb_final_100.ubj")
        scaler = joblib.load("scaler_final_100.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Model files not found: {e}")
        st.stop()

model, scaler = load_model()

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
    if len(sequence) < 5 or len(sequence) > 49:
        return None, f"Length must be 5-49 AA (got {len(sequence)})."
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
    contributions = np.asarray(
        model.get_booster().predict(
            xgb.DMatrix(features_scaled),
            pred_contribs=True,
            approx_contribs=False,
        ),
        dtype=float,
    )
    if contributions.shape != (1, len(FEATURE_NAMES) + 1):
        raise ValueError(f"Unexpected TreeSHAP contribution shape: {contributions.shape}")
    shap_vals = contributions[0, :-1]
    expected = float(contributions[0, -1])
    exp = shap.Explanation(
        values=shap_vals,
        base_values=expected,
        data=features_scaled[0],
        feature_names=FEATURE_NAMES,
    )
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title("Local SHAP explanation - Descriptor contributions")
    st.pyplot(fig)
    plt.close()
    st.caption("SHAP contributions are displayed on the model's raw log-odds scale. Feature values shown in the plot are standardized values used as model inputs.")

def align_sequences(seq1, seq2):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(seq1, seq2)[0]
    s1_aligned, s2_aligned = str(alignment[0]), str(alignment[1])

    score = aligner.score(seq1, seq2)
    similarity = max(score / max(len(seq1), len(seq2)), 0)

    html = '<div style="font-family: monospace; font-size: 14px; line-height: 1.4;">'
    html += '<div style="margin-bottom: 4px;">'
    for a, b in zip(s1_aligned, s2_aligned):
        if a == b and a != '-':
            html += f'<span style="background-color: #c8e6c9; padding: 0 2px;">{a}</span>'
        else:
            html += f'<span style="background-color: #ffcdd2; color: #c62828; padding: 0 2px;">{a}</span>'
    html += '</div><div style="margin-bottom: 4px;">'
    for a, b in zip(s1_aligned, s2_aligned):
        html += '|' if (a == b and a != '-') else ' '
    html += '</div><div>'
    for a, b in zip(s1_aligned, s2_aligned):
        if a == b and a != '-':
            html += f'<span style="background-color: #c8e6c9; padding: 0 2px;">{b}</span>'
        else:
            html += f'<span style="background-color: #ffcdd2; color: #c62828; padding: 0 2px;">{b}</span>'
    html += '</div></div>'
    return html, similarity

def find_most_similar(input_seq, ref_seqs, ref_labels):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    best_ratio = 0
    best_seq = None
    best_label = None
    candidates = [(s, l) for s, l in zip(ref_seqs, ref_labels) if abs(len(s) - len(input_seq)) <= 10]
    for seq, label in candidates:
        score = aligner.score(input_seq, seq)
        similarity = score / max(len(input_seq), len(seq))
        if similarity > best_ratio:
            best_ratio = similarity
            best_seq = seq
            best_label = "Active" if label == 1 else "Inactive"
    return best_seq, best_label, best_ratio

def get_confidence_level(similarity_ratio):
    # Thresholds and accuracy figures derived descriptively from held-out evaluation
    # (n=128; see accompanying article, Results section)
    if similarity_ratio >= 0.8:
        return "Very high", "🟢", "Peptides with a training-set similarity score of at least 0.80 were correctly classified 93.1% of the time in held-out evaluation (n=29)."
    elif similarity_ratio >= 0.7:
        return "High", "🟡", "Peptides with a training-set similarity score from 0.70 to below 0.80 were correctly classified 87.5% of the time in held-out evaluation (n=16)."
    elif similarity_ratio >= 0.3:
        return "Moderate", "🟠", "Peptides with a training-set similarity score from 0.30 to below 0.70 were correctly classified 66.7% of the time in held-out evaluation (n=63)."
    else:
        return "Low", "🔴", "Peptides with a training-set similarity score below 0.30 were correctly classified 50.0% of the time in held-out evaluation (n=20)."

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
        - **Input**: Peptide sequence (5–49 amino acids, uppercase letters A–Y).
        - **Descriptors**: 33 features including amino acid composition, length, net charge, hydrophobicity, and 10 global descriptors (MW, pI, aliphatic index, Boman index, etc.) computed with `modlAMP`.  
        - **Model**: XGBoost classifier trained on a quality-filtered, exact-length-matched dataset of 1,278 peptides (639 active, 639 inactive).
        - **Output**: Probability of being active (0–100%) and a binary prediction (Active/Inactive), plus a local SHAP explanation and similarity search.
        """)

    # Key results (held-out test, 90/10 split model — reported in the article)
    st.subheader("📊 Model performance (held-out test)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC", "0.810")
    col2.metric("F1-score", "0.715")
    col3.metric("Accuracy", "0.727")
    col4.metric("Precision", "0.746")
    col5.metric("Recall", "0.688")
    st.markdown("Held-out evaluation on 10% of the data (128 peptides not used to fit the evaluation model). The model deployed in this app was subsequently retrained on 100% of the dataset; the metrics above come from the 90/10 evaluation reported in the article.")

    with st.expander("⚠️ Limitations", expanded=False):
        st.markdown("""
        - **Short peptides (<10 AA)**: Predictions are less reliable due to underrepresentation in the training set.  
        - **General activity only**: The model does not distinguish between Gram-positive, Gram-negative, or fungal targets.  
        - **No MIC prediction**: The model outputs only the probability of activity, not the minimal inhibitory concentration.  
        - **Sequence length**: Only peptides between 5 and 49 amino acids, using standard L-amino acids, are accepted.
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
    st.markdown("Enter a peptide sequence (5-49 AA, uppercase).")

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
                best_seq, best_label, best_ratio = find_most_similar(sequence, seq_ref, labels_ref)
                st.markdown("---")
                st.subheader("🔍 Similarity with known peptides")
                conf_level, conf_icon, conf_explanation = get_confidence_level(best_ratio)
                st.info(f"{conf_icon} **Similarity-based evidence level:** {conf_level} (based on sequence similarity with training set)")
                st.caption(conf_explanation)

                if np.isclose(best_ratio, 1.0):
                    st.success(f"**Identical to known peptide** (sequence similarity = {best_ratio:.1%})")
                    st.markdown(f"Known peptide: `{best_seq}` ({best_label})")
                    align_html, _ = align_sequences(sequence, best_seq)
                    st.markdown(align_html, unsafe_allow_html=True)
                elif best_ratio >= 0.7:
                    st.info(f"**Similar to known peptide** (sequence similarity = {best_ratio:.1%})")
                    st.markdown(f"Known peptide: `{best_seq}` ({best_label})")
                    align_html, _ = align_sequences(sequence, best_seq)
                    st.markdown(align_html, unsafe_allow_html=True)
                else:
                    st.info(f"No highly similar peptide found (best similarity = {best_ratio:.1%}). Prediction relies on general patterns.")

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

    st.markdown("### Held-out test results")
    metrics = {"AUC": 0.8101, "F1-score": 0.7154, "Accuracy": 0.7266, "Precision": 0.7458, "Recall": 0.6875, "MCC": 0.4545}
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
    - **Total peptides**: 1,278 (639 active, 639 inactive), exactly matched by peptide length
    - **Length range**: 5–49 amino acids (mean 15.94 AA in both classes)
    - **Descriptors**: 33 features (10 global, 20 AA composition, length, net charge, hydrophobicity)  
    - **Model**: XGBoost (n_estimators=200, max_depth=12, learning_rate=0.1, subsample=1.0, colsample_bytree=0.7, gamma=0, reg_lambda=10, reg_alpha=0)
    - **Deployment note**: the model used in this app was retrained on 100% of the dataset above after evaluation; metrics reported here come from the held-out 10% evaluation of the corresponding model configuration trained on the 90% development set.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("XGBoost – Held-out AUC = 0.810")
