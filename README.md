# AMP-Predictor

AMP-Predictor is a Streamlit application that estimates general antimicrobial activity for peptide sequences containing 5–46 standard amino acids. It uses a Random Forest classifier trained with 33 sequence-derived descriptors.

## Live application

https://amp-predictor.streamlit.app/

## Model and evaluation

The A9 production model was trained on 1,196 unique peptides (598 active and 598 inactive) after source-controlled curation, exact-length matching, similarity-grouped splitting, model selection, held-out evaluation, and external validation. Within each retained source, active and inactive counts are equal.

The selected configuration trained on the 1,072-peptide development partition achieved the following results on the 124-peptide held-out set:

- ROC AUC: 0.8689
- Average precision: 0.8570
- F1-score: 0.7500
- Accuracy: 0.7581
- Precision: 0.7759
- Recall: 0.7258
- Matthews correlation coefficient: 0.5172

The held-out set shared no exact sequence or similarity group with development data; maximum cross-partition normalized global-alignment similarity was 0.7778.

External performance varied by source. The model achieved an AUC of 0.7219 on a small 26-peptide DRAMP4 binary cohort, detected 40 of 42 ACS 2026 positives, and detected 5 of 10 recent APD6 positives. These estimates should be interpreted with their reported uncertainty and cohort sizes.

## Dataset curation

The production dataset combines decisive active and inactive sequences from DBAASP and CAMP_R4. Exact cross-source overlaps, cross-source label conflicts, explicitly annotated CAMP modifications, and mixed-label similarity clusters were excluded. Active and inactive peptides were matched at each exact sequence length separately within each source.

## Repository files

- `app.py`: Streamlit application.
- `model_random_forest_final_100.pkl`: production Random Forest.
- `scaler_final_100.pkl`: fitted 33-feature scaler.
- `peptide_features_source_controlled.csv`: complete source-controlled modeling dataset used for production training and similarity search.
- `modlamp/`: locally included modlAMP 4.3.0 modules required for descriptor calculation.
- `THIRD_PARTY_NOTICES.md`: attribution and license information.
- `requirements.txt`: pinned deployment dependencies.

## Interpretation and limitations

Predictions indicate general antimicrobial activity and do not identify a target organism or estimate a minimum inhibitory concentration. Similarity categories provide descriptive context and are not calibrated probabilities of correctness. Local feature explanations use TreeSHAP contributions on the active-class probability scale. Predictions require experimental validation.

## Reference

- Müller, A. T. et al. modlAMP: Python for antimicrobial peptides. *Bioinformatics* 2017, 33, 2753–2755. https://doi.org/10.1093/bioinformatics/btx285
- Associated AMP-Predictor article: citation to be added after publication.
