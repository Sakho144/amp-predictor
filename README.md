# AMP-Predictor

AMP-Predictor is a Streamlit application that predicts general antimicrobial activity for peptide sequences containing 5–50 standard amino acids. It uses an XGBoost classifier trained with 33 sequence-derived descriptors.

## Live application

https://amp-predictor.streamlit.app/

## Model and evaluation

The production model was trained on the complete recurated, exact-length-matched dataset of 1,278 unique peptides (639 active and 639 inactive) after model selection and held-out evaluation were completed. The active and inactive classes have identical peptide-length distributions.

The corresponding configuration trained on the 1,150-peptide development set achieved the following results on the independent 128-peptide held-out set:

- ROC AUC: 0.8101
- F1-score: 0.7154
- Accuracy: 0.7266
- Precision: 0.7458
- Recall: 0.6875
- Matthews correlation coefficient: 0.4545

The deployed XGBoost configuration uses 200 estimators, maximum depth 12, learning rate 0.1, subsample 1.0, column subsampling 0.7, gamma 0, L1 regularization 0, and L2 regularization 10.

## Dataset curation

Active peptides were obtained from APD3 and BIOPEP-UWM. Inactive peptides were selected from DBAASP using unambiguous MIC or IC50 evidence after unit harmonization. Censored, indeterminate, conflicting, and cross-source conflicting records were excluded. The final classes were matched exactly by sequence length before the development and held-out sets were created.

## Repository files

- `app.py`: Streamlit application.
- `model_xgb_final_100.ubj`: native XGBoost production model loaded by the application.
- `model_xgb_final_100.pkl`: Joblib copy of the same production model for reproducibility.
- `scaler_final_100.pkl`: fitted 33-feature scaler.
- `peptide_features_length_balanced.csv`: complete public matched dataset, including sequences, descriptors, identifiers, and activity labels; also used for similarity search.
- `modlamp/`: locally included modlAMP 4.3.0 modules required for descriptor calculation.
- `THIRD_PARTY_NOTICES.md`: attribution and license information for included third-party code.
- `requirements.txt`: pinned deployment dependencies.

## Requirements and local use

The validated runtime uses Python 3.10.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application computes the descriptors with the included modlAMP 4.3.0 code. Its packaging metadata is intentionally excluded because it requires an unavailable historical MySQL connector that is unrelated to descriptor calculation.

## Interpretation

Predictions indicate general antimicrobial activity and do not identify a target organism or estimate a minimum inhibitory concentration. Similarity-based evidence levels summarize held-out accuracy strata and are not calibrated probabilities. Predictions require experimental validation.

Local feature explanations use exact XGBoost TreeSHAP contributions on the raw model margin (log-odds) scale. Descriptor values displayed in the waterfall plot are standardized model inputs.

## References

- Müller, A. T. et al. modlAMP: Python for antimicrobial peptides. *Bioinformatics* 2017, 33, 2753–2755. https://doi.org/10.1093/bioinformatics/btx285
- Associated AMP-Predictor article: citation to be added after publication.
