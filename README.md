# AMP-Predictor

Machine learning web application for predicting antimicrobial activity of food-derived peptides (5–50 amino acids), using an XGBoost classifier trained on a length-bias-corrected dataset.

## Live application
https://amp-predictor.streamlit.app/

## Files
- `app.py` — Streamlit application
- `model_xgb_final_100.pkl` — trained XGBoost model (100% of the final dataset)
- `scaler_final_100.pkl` — associated feature scaler
- `peptide_features_length_balanced.csv` — reference dataset for similarity search
- `requirements.txt` — pinned Python dependencies

## Requirements
- **Python 3.10** (required — newer versions, e.g. 3.13+, break scikit-learn==1.3.0 and SHAP compatibility)
- See `requirements.txt` for exact package versions

## Local installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reference
Associated article: [titre de l'article, à compléter une fois publié]
