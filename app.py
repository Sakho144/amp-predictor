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
from Bio import Align
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="AMP-Predictor", page_icon="🧬", layout="wide")

# ---------- CHARGEMENT MODÈLE ET SCALER ----------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_random_forest_final_100.pkl")
        scaler = joblib.load("scaler_final_100.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Model files not found: {e}")
        st.stop()

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

model, scaler = load_model()
shap_explainer = load_shap_explainer()

# ---------- CHARGEMENT DU DATASET DE RÉFÉRENCE ----------
@st.cache_data
def load_reference_data():
    df = pd.read_csv("peptide_features_source_controlled.csv")
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
    if len(sequence) < 5 or len(sequence) > 46:
        return None, f"Length must be 5-46 AA (got {len(sequence)})."
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
    values = shap_explainer.shap_values(features_scaled, check_additivity=False)
    if isinstance(values, list):
        shap_vals = np.asarray(values[1][0], dtype=float)
    else:
        values = np.asarray(values, dtype=float)
        if values.ndim == 3 and values.shape[2] == 2:
            shap_vals = values[0, :, 1]
        elif values.ndim == 2:
            shap_vals = values[0]
        else:
            raise ValueError(f"Unexpected TreeSHAP shape: {values.shape}")
    expected_values = np.asarray(shap_explainer.expected_value, dtype=float).reshape(-1)
    expected = float(expected_values[1] if expected_values.size > 1 else expected_values[0])
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
    st.caption("SHAP contributions are displayed on the active-class probability scale. Feature values shown in the plot are standardized model inputs.")

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
    # Descriptive similarity context from the leakage-resistant A9 held-out split.
    # These categories are not calibrated probabilities of prediction correctness.
    if similarity_ratio >= 0.8:
        return "Very high", "🟢", "No held-out peptide fell in this similarity range because groups with similarity of at least 0.80 were kept in the same partition."
    elif similarity_ratio >= 0.7:
        return "High", "🟡", "Held-out accuracy in this range was 85.7% (12/14 peptides)."
    elif similarity_ratio >= 0.5:
        return "Moderate to high", "🟠", "Held-out accuracy in this range was 75.8% (25/33 peptides)."
    elif similarity_ratio >= 0.3:
        return "Moderate", "🟠", "Held-out accuracy in this range was 76.6% (36/47 peptides)."
    else:
        return "Low", "🔴", "Held-out accuracy in this range was 70.0% (21/30 peptides)."

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
        - **Input**: Peptide sequence (5–46 standard amino acids, using only ACDEFGHIKLMNPQRSTVWY).
        - **Descriptors**: 33 features including amino acid composition, length, net charge, hydrophobicity, and 10 global descriptors (MW, pI, aliphatic index, Boman index, etc.) computed with `modlAMP`.  
        - **Model**: Random Forest classifier trained on a source-controlled, exact-length-matched dataset of 1,196 peptides (598 active, 598 inactive).
        - **Output**: Probability of being active (0–100%) and a binary prediction (Active/Inactive), plus a local SHAP explanation and similarity search.
        """)

    # Key results (held-out test, 90/10 split model — reported in the article)
    st.subheader("📊 Model performance (held-out test)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC", "0.869")
    col2.metric("F1-score", "0.750")
    col3.metric("Accuracy", "0.758")
    col4.metric("Precision", "0.776")
    col5.metric("Recall", "0.726")
    st.markdown("Held-out evaluation on 124 peptides separated from development data by sequence-similarity groups. The model deployed in this app was subsequently retrained on 100% of the dataset; the metrics above come from the 90/10 evaluation reported in the article.")

    with st.expander("⚠️ Limitations", expanded=False):
        st.markdown("""
        - **Short peptides (<10 AA)**: Predictions are less reliable due to underrepresentation in the training set.  
        - **General activity only**: The model does not distinguish between Gram-positive, Gram-negative, or fungal targets.  
        - **No MIC prediction**: The model outputs only the probability of activity, not the minimal inhibitory concentration.  
        - **Sequence length**: Only peptides between 5 and 46 amino acids, using standard L-amino acids, are accepted.
        - **External validation**: Performance varied across sources (DRAMP4 AUC 0.722; ACS sensitivity 95.2%; APD6 sensitivity 50.0%).
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
    st.markdown("Enter a peptide sequence (5-46 standard amino acids, uppercase).")

    if 'seq_input' not in st.session_state:
        st.session_state.seq_input = ""

    sequence = st.text_area("Peptide sequence", height=100, value=st.session_state.seq_input)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load example (Magainin-2)"):
            st.session_state.seq_input = "GIGKFLHSAKKFGKAFVGEIMNS"
            st.rerun()
    with col2:
        if st.button("Load example (Poly-Alanine)"):
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
    metrics = {"AUC": 0.8689, "F1-score": 0.7500, "Accuracy": 0.7581, "Precision": 0.7759, "Recall": 0.7258, "MCC": 0.5172}
    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.dataframe(df_metrics, width="stretch", hide_index=True)

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
    - **Total peptides**: 1,196 (598 active, 598 inactive), balanced within source and exactly matched by peptide length
    - **Length range**: 5–46 amino acids (mean 14.32 AA in both classes)
    - **Descriptors**: 33 features (10 global, 20 AA composition, length, net charge, hydrophobicity)  
    - **Model**: Random Forest (600 trees, maximum depth 20, minimum split size 5, square-root feature sampling, balanced-subsample class weights)
    - **Deployment note**: the model used in this app was retrained on 100% of the dataset above after evaluation; metrics reported here come from the held-out 10% evaluation of the corresponding model configuration trained on the 90% development set.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Random Forest – Held-out AUC = 0.869")
