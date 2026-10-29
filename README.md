Emotion Recognition Using EEG Signals (DEAP, MATLAB)

A two-phase study of EEG-based emotion recognition on the DEAP dataset: a rule-based frontal asymmetry baseline, followed by a leakage-safe, imbalance-corrected gradient-boosted ensemble. Built entirely in MATLAB as a summer internship project at JSS University Noida (Dept. of ECE), July 2026.

Highlights
Phase 1 (rule-based, no ML): Frontal Alpha Asymmetry (valence) and log Beta/Alpha ratio (arousal) with per-subject z-scoring. Result: 47.6-51.7%, i.e. chance level. A well-diagnosed null result that motivated Phase 2.
Phase 2 (ML): 518 features across all 32 channels, RUSBoost classifier, subject-grouped 8-fold cross-validation, pruned to the Top-25 features.
Final result (subject-independent): Balanced accuracy 59.1% (valence) and 60.0% (arousal); Macro-F1 0.57 and 0.58.
Deep learning (CNN/GNN): explored, but less sample-efficient than the boosted ensemble on a 32-subject cohort, so not selected.
Dataset

DEAP: 32 subjects x 40 one-minute music-video trials, 32 EEG channels, 128 Hz, 1,280 trials total. Valence and arousal are binarised at 4.5.

Pipeline
Load all subject files and remove the 3 s pre-trial baseline.
8th-order zero-phase IIR band-pass filtering (delta, theta, alpha, beta, gamma).
Phase 1: FAA over four frontal pairs (Fp1/Fp2, AF3/AF4, F3/F4, F7/F8) and log Beta/Alpha across 8 frontal channels.
Phase 2: extract the 518-dimensional feature set (below), z-score per subject.
Train RUSBoost with subject-grouped 8-fold cross-validation.
Rank features by importance and retrain on the Top-25.
Feature families (518 total)
Family	Dimensions
Differential Entropy (DE)	160
Relative Band Power (RP)	160
Higuchi Fractal Dimension (HFD)	32
Hjorth parameters	96
Differential Asymmetry (DASM)	70
Key Findings
Majority-class trap: the standard DEAP threshold gives 79.0% Low for valence and 76.7% Low for arousal. Uncorrected LogitBoost reached about 78% raw accuracy by predicting mostly the majority class. RUSBoost with Balanced Accuracy and Macro-F1 fixes this.
Data leakage: whole subjects are assigned to folds so no subject appears in both train and test.
Curse of dimensionality: the full 518 features gave about 51-53% balanced accuracy; the Top-25 gave about 60%.
Beyond the frontal lobe: top predictors include parietal, occipital and central-parietal channels, and span DASM, DE and RP families.
Target	Full set Bal. Acc.	Top-25 Bal. Acc.	Top-25 Macro-F1
Valence	51.4%	59.1%	0.57
Arousal	53.0%	60.0%	0.58
Requirements
MATLAB
Signal Processing Toolbox (filtering, Welch PSD)
Statistics and Machine Learning Toolbox (RUSBoost, feature importance, confusion matrices)
Deep Learning Toolbox (only for the optional CNN exploration)
Limitations
8-fold CV has roughly 5-6 point fold-to-fold standard deviation; Leave-One-Subject-Out would tighten this.
Gamma-band predictors may be influenced by EMG artefacts (no ICA-based rejection yet).
Single-feature rankings are unstable among correlated features; family-level conclusions are more robust.
The Phase 1 null result applies to this specific 2-feature frontal rule, not to FAA in general.
Future Work
Leave-One-Subject-Out validation
ICA-based EMG artefact removal
Stability selection for feature ranking
CNN/GNN on larger corpora (DEAP + SEED / MAHNOB-HCI)
Continuous valence/arousal regression
Real-time deployment feasibility
