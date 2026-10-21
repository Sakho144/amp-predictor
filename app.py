#!/usr/bin/env python3
"""AMP-Predictor: the manuscript Random Forest application."""

from pathlib import Path
from html import escape

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from Bio import Align
from modlamp.descriptors import GlobalDescriptor

ROOT = Path(__file__).resolve().parent
FEATURES = [f"global_{i}" for i in range(10)] + [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"] + [
    "length", "net_charge", "hydrophobicity"]
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
EXTERNAL_STUDIES = pd.DataFrame([
    ("Jiang 2026", 20, 4, 16, 0.828, 1.000, 0.500),
    ("Liu 2026", 9, 4, 5, 0.650, 1.000, 0.200),
    ("Muletz-Wolz 2025", 8, 2, 6, 1.000, 1.000, 1.000),
    ("Huang 2026, low similarity", 99, 26, 73, 0.785, 0.962, 0.233),
    ("Irvine 2023, low similarity", 15, 8, 7, 0.804, 1.000, 0.286),
], columns=["Study", "n", "Active", "Inactive", "ROC-AUC", "Sensitivity", "Specificity"])
AMINO_ACID_GUIDE_HTML = """
<div class="aa-guide" aria-live="polite">
  <div class="aa-guide-title">Sequence guide</div>
  <div class="aa-guide-current">Type a sequence to see residue names.</div>
  <div class="aa-guide-letters" aria-label="Amino-acid sequence"></div>
  <div class="aa-guide-hint">Select a letter to see its full amino-acid name.</div>
</div>
"""
AMINO_ACID_GUIDE_CSS = """
.aa-guide {
  border: 1px solid rgba(128, 128, 128, .28);
  border-radius: .65rem;
  padding: .7rem .85rem;
  margin: .15rem 0 .65rem;
  background: var(--st-secondary-background-color, #f6f8fa);
  font-family: var(--st-font, sans-serif);
}
.aa-guide-title { font-weight: 650; margin-bottom: .15rem; }
.aa-guide-current { min-height: 1.4rem; }
.aa-guide-letters { display: flex; flex-wrap: wrap; gap: .25rem; margin-top: .35rem; }
.aa-guide-letter {
  min-width: 1.55rem;
  border: 1px solid rgba(128, 128, 128, .35);
  border-radius: .32rem;
  padding: .12rem .24rem;
  background: var(--st-background-color, white);
  color: inherit;
  cursor: pointer;
  font: inherit;
  font-weight: 600;
}
.aa-guide-letter:focus-visible { outline: 2px solid var(--st-primary-color, #ff4b4b); }
.aa-guide-letter-invalid { color: #c53030; border-color: #c53030; }
.aa-guide-hint { font-size: .82rem; opacity: .75; margin-top: .25rem; }
"""
AMINO_ACID_GUIDE_JS = """
export default function({ parentElement }) {
  const root = parentElement.querySelector('.aa-guide');
  const current = root.querySelector('.aa-guide-current');
  const letters = root.querySelector('.aa-guide-letters');
  const names = {
    A: 'Alanine', C: 'Cysteine', D: 'Aspartic acid', E: 'Glutamic acid',
    F: 'Phenylalanine', G: 'Glycine', H: 'Histidine', I: 'Isoleucine',
    K: 'Lysine', L: 'Leucine', M: 'Methionine', N: 'Asparagine',
    P: 'Proline', Q: 'Glutamine', R: 'Arginine', S: 'Serine',
    T: 'Threonine', V: 'Valine', W: 'Tryptophan', Y: 'Tyrosine'
  };
  let input = null;

  function describe(letter) {
    return names[letter] ? `${letter} — ${names[letter]}` :
      `${letter} — non-standard residue`;
  }
  function render() {
    const sequence = (input?.value || '').toUpperCase().replace(/\\s/g, '');
    letters.replaceChildren();
    if (!sequence) {
      current.textContent = 'Type a sequence to see residue names.';
      return;
    }
    current.textContent = describe(sequence[sequence.length - 1]);
    for (const letter of sequence.slice(0, 46)) {
      const token = document.createElement('button');
      token.type = 'button';
      token.className = 'aa-guide-letter' + (names[letter] ? '' : ' aa-guide-letter-invalid');
      token.textContent = letter;
      token.title = describe(letter);
      token.setAttribute('aria-label', describe(letter));
      token.addEventListener('click', () => { current.textContent = describe(letter); });
      letters.appendChild(token);
    }
    if (sequence.length > 46) {
      const overflow = document.createElement('span');
      overflow.textContent = `… ${sequence.length - 46} more`;
      letters.appendChild(overflow);
    }
  }
  function bind() {
    const next = document.querySelector('textarea[aria-label="Peptide sequence"]') ||
      document.querySelector('textarea');
    if (!next || next === input) return;
    if (input) input.removeEventListener('input', render);
    input = next;
    input.addEventListener('input', render);
    render();
  }
  bind();
  const observer = new MutationObserver(bind);
  observer.observe(document.body, { childList: true, subtree: true });
  return () => {
    observer.disconnect();
    if (input) input.removeEventListener('input', render);
  };
}
"""
amino_acid_guide = st.components.v2.component(
    "amp_amino_acid_guide",
    html=AMINO_ACID_GUIDE_HTML,
    css=AMINO_ACID_GUIDE_CSS,
    js=AMINO_ACID_GUIDE_JS,
)
HYDRO_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
    "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
    "N": -3.5, "K": -3.9, "R": -4.5,
}
def compute_features(sequence: str) -> np.ndarray:
    sequence = sequence.strip().upper()
    if not 5 <= len(sequence) <= 46:
        raise ValueError(f"Peptide length must be 5–46 residues; received {len(sequence)}.")
    bad = set(sequence) - AMINO_ACIDS
    if bad:
        raise ValueError(f"Non-standard residue(s): {', '.join(sorted(bad))}.")
    descriptor = GlobalDescriptor([sequence])
    descriptor.calculate_all()
    globals_ = np.asarray(descriptor.descriptor[0], dtype=float).reshape(-1)
    if globals_.size != 10:
        raise RuntimeError("Unexpected number of modlAMP global descriptors")
    composition = [sequence.count(aa) / len(sequence) for aa in "ACDEFGHIKLMNPQRSTVWY"]
    charge = sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")
    hydrophobicity = sum(HYDRO_SCALE[aa] for aa in sequence) / len(sequence)
    values = np.concatenate([globals_, composition, [len(sequence), charge, hydrophobicity]])
    if values.size != 33 or not np.isfinite(values).all():
        raise RuntimeError("Invalid descriptor result")
    return values.reshape(1, -1)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "ANF_random_forest_1178.joblib")
    reference = pd.read_csv(ROOT / "ANF_training_peptides_1178.csv")
    if reference.sequence.nunique() != 1178 or len(reference) != 1178:
        raise RuntimeError("ANF training reference must contain 1,178 unique peptides")
    if set(reference.activity_label) != {0, 1} or not set(FEATURES) <= set(reference.columns):
        raise RuntimeError("Invalid production reference table")
    if (list(model.named_steps) != ["standardscaler", "randomforestclassifier"] or
            list(model.classes_) != [0, 1] or model.n_features_in_ != 33):
        raise RuntimeError("Unexpected Random Forest class order or feature count")
    explainer = shap.TreeExplainer(model.named_steps["randomforestclassifier"])
    return model, reference, explainer


def score_sequence(sequence: str, model) -> tuple[np.ndarray, float]:
    values = compute_features(sequence)
    score = float(model.predict_proba(values)[0, 1])
    if not 0 <= score <= 1:
        raise RuntimeError("Invalid active-class score")
    return values, score


def local_explanation(values: np.ndarray, score: float, model, explainer):
    standardized = model.named_steps["standardscaler"].transform(values)
    raw = explainer(standardized)
    all_contributions = np.asarray(raw.values, dtype=float)
    all_baselines = np.asarray(raw.base_values, dtype=float)
    if all_contributions.shape == (1, 33, 2):
        contributions = all_contributions[0, :, 1]
        baseline = float(all_baselines[0, 1])
    elif all_contributions.shape == (1, 33):
        contributions = all_contributions[0]
        baseline = float(all_baselines.reshape(-1)[0])
    else:
        raise RuntimeError(f"Unexpected TreeSHAP shape: {all_contributions.shape}")
    if abs(baseline + contributions.sum() - score) > 1e-8:
        raise RuntimeError("TreeSHAP values do not add up to the active-class score")
    # Render the validated SHAP values on a figure owned by this call. This
    # avoids pyplot's shared-axis state, which can superimpose unrelated ticks.
    from matplotlib.figure import Figure

    ranked = np.argsort(np.abs(contributions))[::-1]
    shown = ranked[:14]
    rows = [(f"{FEATURES[i]} ({standardized[0, i]:+.2f})", float(contributions[i]))
            for i in shown]
    rows.append(("Other 19 features", float(contributions[ranked[14:]].sum())))
    names, effects = zip(*rows)
    figure = Figure(figsize=(11.5, 8.5), dpi=120, constrained_layout=True)
    ax = figure.subplots()
    positions = np.arange(len(rows))
    colors = ["#ff0051" if value >= 0 else "#0788ea" for value in effects]
    ax.barh(positions, effects, color=colors, height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="#59636e", linewidth=1)
    largest = max(abs(value) for value in effects)
    ax.set_xlim(-largest * 1.45, largest * 1.45)
    for position, effect in enumerate(effects):
        offset = largest * 0.035
        ax.text(effect + (offset if effect >= 0 else -offset), position,
                f"{effect:+.2f}", va="center",
                ha="left" if effect >= 0 else "right", fontsize=9)
    ax.set_xlabel("Contribution to active-class model score (positive favors Active)")
    ax.set_title("Local SHAP explanation — descriptor contributions\n"
                 f"Baseline {baseline:.3f} + contributions = score {score:.3f}")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    return figure


def closest_reference(sequence: str, reference: pd.DataFrame):
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    candidates = reference.loc[reference.sequence.str.len().sub(len(sequence)).abs() <= 10]
    best = None
    for row in candidates.itertuples(index=False):
        similarity = max(float(aligner.score(sequence, row.sequence)) /
                         max(len(sequence), len(row.sequence)), 0.0)
        if best is None or similarity > best[2]:
            best = (row.sequence, int(row.activity_label), similarity)
    return best


def alignment_markup(query: str, reference_sequence: str) -> str:
    """Reproduce the original colored, aligned-sequence display."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query, reference_sequence)[0]
    first, second = str(alignment[0]), str(alignment[1])
    first_html = []
    second_html = []
    marks = []
    for left, right in zip(first, second):
        equal = left == right and left != "-"
        style = "background-color:#c8e6c9" if equal else "background-color:#ffcdd2;color:#c62828"
        first_html.append(f'<span style="{style};padding:0 2px">{escape(left)}</span>')
        second_html.append(f'<span style="{style};padding:0 2px">{escape(right)}</span>')
        marks.append("|" if equal else " ")
    return ("<div style='font-family:monospace;font-size:14px;line-height:1.4;"
            "overflow-x:auto;white-space:nowrap'>"
            f"<div>{''.join(first_html)}</div>"
            f"<div>{''.join(marks)}</div>"
            f"<div>{''.join(second_html)}</div></div>")


st.set_page_config(page_title="AMP-Predictor", page_icon="🧬", layout="wide")
st.sidebar.title("🧬 AMP-Predictor")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🧪 Prediction", "📊 Model performance", "🔬 External validation"])

try:
    MODEL, REFERENCE, EXPLAINER = load_artifacts()
except Exception as error:
    st.error(f"The validated model files could not be loaded: {error}")
    st.stop()

if page == "🏠 Home":
    st.title("AMP-Predictor: Antimicrobial Peptide Activity Prediction")
    st.markdown("---")

    with st.expander("📖 About this tool", expanded=True):
        st.markdown("""
        **AMP-Predictor** is a research tool for screening peptide sequences for
        general antimicrobial activity. It helps prioritize candidates for
        experimental testing; it does not replace those tests.
        """)

    with st.expander("⚙️ How it works", expanded=False):
        st.markdown("""
        - **Input**: 5–46 letters from the 20 standard amino acids.
        - **Descriptors**: 33 sequence-derived features, including composition,
          length, charge, hydrophobicity and 10 global descriptors. The global
          mass, pH-charge and pI descriptors use an assumed C-terminal amide;
          the actual terminal chemistry of source peptides is not always known.
        - **Model**: Random Forest fitted on 1,178 unique peptides from DBAASP
          and CAMP R4 (589 active and 589 inactive), with source-specific
          exact-length matching and grouped cross-validation.
        - **Label**: Activity follows the study's operational MIC rule at 100 µM.
        - **Output**: Active/inactive classification, active-class model score,
          nearby training sequence and a peptide-specific SHAP explanation.
        """)

    st.subheader("📊 Grouped cross-validation")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean ROC-AUC", "0.796")
    col2.metric("F1-score", "0.732")
    col3.metric("Accuracy", "0.733")
    col4.metric("Precision", "0.736")
    col5.metric("Recall", "0.728")
    st.caption("The AUC is the mean of five grouped outer folds. The other values pool their out-of-fold predictions at the fixed 0.5 threshold. The application model was subsequently fitted on all 1,178 peptides and has no separate held-out estimate.")

    st.caption("Results from published peptide studies vary by source. See External validation for the study-by-study results and their limits.")

    with st.expander("⚠️ Limitations", expanded=False):
        st.markdown("""
        - Predicts **general activity under the study's MIC rule**, not activity
          against a named organism or a numerical MIC.
        - Accepts only 5–46 standard residue letters; D-residues and chemical
          modifications are not represented by a letter-only input.
        - Performance varies across external studies; see the dedicated
          External validation page for results and study-specific limitations.
        - A model score is not an experimentally measured MIC or a calibrated
          probability for the individual peptide. Confirm prioritized candidates experimentally.
        """)

    with st.expander("🤝 Collaboration & Contact", expanded=False):
        st.markdown("""
        To discuss collaboration or testing on new peptide datasets, contact:
        **mahamadou.sakho@etu.uae.ac.ma**
        """)

elif page == "🧪 Prediction":
    st.title("Predict antimicrobial activity")
    st.markdown("Enter a peptide sequence (5–46 standard amino-acid letters).")
    sequence = st.text_area("Peptide sequence", height=100).strip().upper()
    amino_acid_guide(key="residue_guide")

    if st.button("Predict", type="primary"):
        try:
            features, score = score_sequence(sequence, MODEL)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        label = "Active" if score >= 0.5 else "Inactive"
        st.success(f"### Prediction: **{label}** (active-class score {score:.2%})")
        st.progress(score)
        st.caption("Classification uses the fixed 50% model-score threshold. The score is uncalibrated and is not a measured probability of activity for this peptide.")
        if len(sequence) <= 8:
            st.warning("In a separate study of 6–8-residue peptides, the model frequently classified experimentally inactive peptides as active.")

        st.markdown("---")
        st.subheader("🔍 Similarity with training peptides")
        nearest = closest_reference(sequence, REFERENCE)
        if nearest is None:
            st.info("No training peptide within ±10 residues of the query length.")
        else:
            ref_sequence, ref_label, similarity = nearest
            ref_text = "Active" if ref_label == 1 else "Inactive"
            st.info(f"**Normalized global alignment score:** {similarity:.1%}")
            st.markdown(f"Training peptide: `{ref_sequence}` ({ref_text})")
            st.markdown(alignment_markup(sequence, ref_sequence), unsafe_allow_html=True)
            st.caption("Compared with training sequences within ±10 residues. Alignment score divided by the longer sequence length; it is not percent identity or prediction confidence.")
            if similarity >= 1 - 1e-12:
                st.warning("This exact sequence is in the training data; its prediction is not an independent validation result.")
            if similarity >= 1 - 1e-12 and ref_label != int(score >= 0.5):
                st.warning("The prediction differs from this sequence's recorded training label; consider the underlying MIC and target context.")

        st.subheader("Calculated properties")
        st.dataframe(pd.DataFrame([
            ("Length (AA)", int(features[0, FEATURES.index("length")])),
            ("Mass descriptor (Da)", round(float(features[0, FEATURES.index("global_1")]), 2)),
            ("Side-chain charge count (K+R−D−E)", round(float(features[0, FEATURES.index("net_charge")]), 2)),
            ("pI descriptor", round(float(features[0, FEATURES.index("global_4")]), 2)),
            ("Hydrophobicity descriptor", round(float(features[0, FEATURES.index("hydrophobicity")]), 4)),
        ], columns=["Property", "Value"]), hide_index=True)
        st.caption("Mass and pI are model descriptors calculated with an assumed C-terminal amide. The actual terminal chemistry of a sequence is not established by its letters alone.")

        st.subheader("SHAP explanation")
        try:
            figure = local_explanation(features, score, MODEL, EXPLAINER)
            st.pyplot(figure)
            plt.close(figure)
        except Exception as error:
            st.warning(f"SHAP plot not available: {error}")
        st.caption("SHAP bars show contributions to the Random Forest active-class score. Displayed feature values are standardized model inputs.")

elif page == "📊 Model performance":
    st.title("Model performance (general activity)")
    st.markdown("### Grouped cross-validation on 1,178 peptides")
    st.dataframe(pd.DataFrame([
        ("Mean outer-fold ROC-AUC", 0.7960),
        ("Mean outer-fold average precision", 0.8108),
        ("Pooled out-of-fold accuracy", 0.7334),
        ("Pooled out-of-fold precision", 0.7358),
        ("Pooled out-of-fold recall", 0.7284),
        ("Pooled out-of-fold F1-score", 0.7321),
    ], columns=["Metric", "Value"]), hide_index=True)
    st.caption("Five grouped outer folds were used for model comparison. At the fixed 0.5 threshold, pooled out-of-fold counts were TN=435, FP=154, FN=160, TP=429. The application model was refitted on all 1,178 peptides; it has no separate held-out result.")

    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
        - **ROC-AUC** measures discrimination across score thresholds.
        - **Precision** is the share of predicted actives that were active in the test.
        - **Recall** is the share of test actives detected at the 50% threshold.
        - **F1-score** combines precision and recall.
        - **MCC** summarizes all four confusion-matrix counts (−1 to +1).
        """)

    st.markdown("---")
    st.markdown("### Training dataset")
    st.markdown("""
    - **1,178 unique peptides**, 589 per class, from DBAASP and CAMP R4.
    - Active and inactive peptides were exactly length-matched within source.
    - **33 sequence-derived descriptors**; Random Forest was selected using
      grouped, nested cross-validation before external scoring.
    - Similarity and publication groups were held together where known.
      A usable publication identifier was unavailable for 892 training entries.
    """)

    with st.expander("🔎 Evaluation details", expanded=False):
        st.markdown("""
        The cross-validation folds kept known similarity, matched-pair and
        publication groups together. The mean fold ROC-AUC was **0.796**.
        The older A9 held-out set is not a new blind test of this repaired model.
        The full-data application model has no independent performance estimate
        of its own. Results on external studies and their limitations appear in
        the External validation page.
        """)

else:
    st.title("External validation and limitations")
    st.markdown("The frozen Random Forest model and its fixed 0.5 threshold were applied to peptides from separate published studies. Results are shown **by study** because assay conditions, target organisms and peptide chemistry differ.")
    st.dataframe(EXTERNAL_STUDIES, hide_index=True)
    st.caption("The descriptive pooled result for Jiang, Liu and Muletz-Wolz was n=37 (10 active, 27 inactive), ROC-AUC 0.852, sensitivity 10/10 and specificity 15/27. It summarizes those three studies; it is not an additional independent dataset.")

    st.markdown("### Positive-only study")
    st.markdown("**ACS 2026:** 40 of 42 peptides were classified active (sensitivity 95.2%). Because the cohort contains no inactive peptides, ROC-AUC and specificity cannot be estimated.")

    with st.expander("Secondary transfer from the earlier SVM data pool", expanded=False):
        st.dataframe(pd.DataFrame([
            ("Exact-novel transfer", 172, 150, 22, 0.778, 0.693, 0.636),
            ("Low similarity, no known shared publication", 107, 92, 15, 0.753, 0.728, 0.467),
        ], columns=["Cohort", "n", "Active", "Inactive", "ROC-AUC", "Sensitivity", "Specificity"]),
            hide_index=True)
        st.caption("These peptides come from the same families of source databases used for training. This is a retrospective transfer analysis, not an independent database validation. The 107-peptide group is a subset of the 172-peptide group; do not add their counts.")

    st.markdown("### What these results show")
    st.markdown("""
    - The model detects many active peptides, but its ability to reject inactive
      peptides differs markedly between studies.
    - In **Huang 2026**, it detected 25/26 active peptides but classified 56/73
      inactive peptides as active. This cohort contains short (6–8 residue),
      C-terminally amidated peptides. Its ROC-AUC was 0.785; sequence length
      alone reached 0.779 in the same cohort.
    - The study challenges are small or concentrated in one study. Their
      within-study results do not establish performance for every organism,
      peptide chemistry or future dataset. The model does not predict MIC.
    """)

    with st.expander("Published sources", expanded=False):
        st.markdown("""
        - [Jiang 2026](https://doi.org/10.1038/s41467-026-71145-0)
        - [Liu 2026](https://doi.org/10.1038/s44386-026-00045-6)
        - [Muletz-Wolz 2025](https://doi.org/10.1038/s41522-025-00837-0)
        - [Huang 2026](https://doi.org/10.1038/s41467-026-69306-2)
        - [Irvine 2023](https://doi.org/10.1371/journal.ppat.1011508)
        """)

st.sidebar.markdown("---")
st.sidebar.caption("Random Forest – grouped-CV mean AUC = 0.796")
