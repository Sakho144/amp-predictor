# AMP-Predictor

Research application for predicting general antimicrobial activity from peptide sequences of 5–46 standard amino-acid letters.

Application: https://amp-predictor.streamlit.app/

## Model

The manuscript model is a Random Forest fitted on 1,178 unique peptides (589 active and 589 inactive), curated from DBAASP and CAMP R4. Active and inactive sequences were length-matched within source. The serialized pipeline contains the fitted StandardScaler and Random Forest together. It uses 33 descriptors and a fixed active-class score threshold of 0.5.

Grouped nested cross-validation gave a mean ROC–AUC of 0.7960 (SD 0.0407) and mean average precision of 0.8108. Pooled out-of-fold counts were TN=435, FP=154, FN=160, TP=429. These describe validation models; fitting the application model on all 1,178 sequences adds no performance estimate. Random Forest did not show a clear advantage over SVM in the paired comparison.

## External evaluation

| Study | Peptides | ROC–AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Jiang 2026 | 20 | 0.828 | 1.000 | 0.500 |
| Liu 2026 | 9 | 0.650 | 1.000 | 0.200 |
| Muletz-Wolz 2025 | 8 | 1.000 | 1.000 | 1.000 |
| Huang 2026, low similarity | 99 | 0.785 | 0.962 | 0.233 |
| Irvine 2023, low similarity | 15 | 0.804 | 1.000 | 0.286 |

Results are study-specific at the unchanged 0.5 threshold. The positive-only ACS analysis detected 40/42 active peptides and cannot estimate specificity or ROC–AUC. Retrospective transfer results are identified separately in the application. These small or specialized datasets do not establish general performance across organisms or peptide chemistries.

## Files

- `app.py`: interface, descriptor calculation, similarity search and TreeSHAP explanations.
- `ANF_random_forest_1178.joblib`: final fitted scaler/classifier pipeline.
- `ANF_training_peptides_1178.csv`: training sequences, labels and 33 descriptors.
- `modlamp/`: required modlAMP 4.3.0 components and original license.
- `requirements.txt`: pinned application dependencies.
- `RELEASE_MANIFEST.json`: artifact hashes and model identity.

The ANF filenames are project identifiers for this Random Forest, not a different algorithm. This deployment repository contains the application and its training reference; it is not a complete archive of all manuscript analyses and supplementary materials.

## Running and deployment

Use Python 3.10, install `requirements.txt`, then run `python -m streamlit run app.py`. In Streamlit Community Cloud, select this repository, branch `main`, entry point `app.py` and Python 3.10 in the deployment settings. Check the build logs after pushing. Changing Git alone does not verify successful deployment.

## Interpretation

Labels use the study's MIC definition at 100 µM. Scores are not calibrated probabilities or MIC estimates and do not identify organism-specific activity. Sequence letters cannot encode terminal modifications or D-amino acids. Global descriptors assume C-terminal amidation at pH 7.4. The normalized alignment score provides sequence context, not prediction confidence. SHAP explains the model score, not biological causation.

## Sources and license notices

- modlAMP: https://doi.org/10.1093/bioinformatics/btx285
- Jiang: https://doi.org/10.1038/s41467-026-71145-0
- Liu: https://doi.org/10.1038/s44386-026-00045-6
- Muletz-Wolz: https://doi.org/10.1038/s41522-025-00837-0
- Huang: https://doi.org/10.1038/s41467-026-69306-2
- Irvine: https://doi.org/10.1371/journal.ppat.1011508

See THIRD_PARTY_NOTICES.md and modlamp/LICENSE for bundled third-party code.
