EEG Emotion Recognition on DEAP: A Leakage-Free, Imbalance-Aware Pipeline (MATLAB)

Subject-independent classification of valence and arousal from 32-channel EEG, built to answer one question honestly: how much of the reported accuracy on DEAP is real signal, and how much is class imbalance and data leakage?
Highlights-
1) Strict subject-independent evaluation. 8-fold cross-validation where whole subjects are held out, so no subject appears in both training and test data.
2)Caught a misleading headline number. With the standard DEAP threshold, 79.0% of valence labels and 76.7% of arousal labels fall in one class. A LogitBoost model scored about 78% accuracy by predicting the majority class almost every time. A majority-class predictor has a balanced accuracy of exactly 50%, so I report Balanced Accuracy and Macro-F1 instead.
3)Feature pruning matters more than model size. Reducing 518 features to the Top-25 raised balanced accuracy from near chance to about 60%.

Results-
Subject-independent, 8-fold subject-grouped CV, RUSBoost:

Target	All 518 features (Bal. Acc.)	Top-25 features (Bal. Acc.)	Top-25 Macro-F1
Valence	51.4%	59.1%	0.57
Arousal	53.0%	60.0%	0.58
Higher figures are common in the literature when subject-dependent or random trial-level splits are used, because those let a model see the same person's EEG during training. This project deliberately uses the harder, more realistic protocol.

Approach
Problem                  	Decision	                                                     Why
Strong class imbalance    RUSBoost (random under-sampling inside boosting)	Stops the model collapsing to                                                                                 the majority class
Inter-subject variability     Per-subject z-scoring of features	      Removes subject-specific    baselines
Risk of leakage	                Subject-grouped folds	                       Gives an honest estimate of                                                                                 performance on a new person
518 correlated features, 32 subjects  Rank with predictorImportance, keep Top-25	Reduces overfitting to                                                                                              noise
Misleading accuracy	Balanced Accuracy, Macro-F1, confusion matrices	       Penalises ignoring the minority                                                                                       class
Feature set (518 per trial): Differential Entropy (160), Relative Band Power (160), Higuchi Fractal Dimension (32), Hjorth parameters (96), Differential Asymmetry (70).

Pipeline: load 32 subjects x 40 trials (128 Hz, 32 channels) -> remove the 3 s pre-trial baseline -> 8th-order zero-phase IIR band-pass filtering -> feature extraction -> subject-wise z-scoring -> RUSBoost with subject-grouped CV -> feature pruning -> evaluation.

Baselines and Exploration
1) Rule-based baseline (no ML): Frontal Alpha Asymmetry for valence and a log Beta/Alpha ratio for arousal. Accuracy stayed at chance (47.6% to 51.7%), which showed that a two-feature frontal rule cannot capture emotion in DEAP and motivated the full-scalp feature approach.
2) Compact 2D CNN (exploratory): Trained on raw 3-second windows (32 channels x 384 samples, 25,600 examples). It did not surpass the RUSBoost model on this 32-subject cohort, which I attribute to the small dataset. Deep models on raw EEG are data-hungry, which is the motivation for pretraining on larger unlabelled EEG corpora.
This was a three-member team project. I implemented the 518-feature extraction pipeline (Differential Entropy, relative band power, Higuchi Fractal Dimension, Hjorth parameters and differential asymmetry), set up the subject-grouped cross-validation with RUSBoost, and ran the exploratory CNN experiments. I also identified that the 78% accuracy of an early model was a majority-class artefact and moved our evaluation to Balanced Accuracy and Macro-F1.
