# -eeg-emotion-recognition-deap-.
clc; clear all; close all;
%% EEG FEATURE EXTRACTION FOR ML (DE + relative band power + HFD + Hjorth + asymmetry)
% -------------------------------------------------------------------------
% Extracts a broad feature set across ALL 32 EEG channels for every trial
% of every subject.
%
% Feature groups (518 features total):
%   DE     - Differential Entropy, 5 bands x 32 channels = 160
%   RP     - Relative band power, 5 bands x 32 channels  = 160
%   HFD    - Higuchi Fractal Dimension, 32 channels       = 32
%   Hjorth - Activity/Mobility/Complexity, 3 x 32 channels = 96
%   DASM   - DE asymmetry (right-left) per band per hemispheric pair,
%            5 bands x 14 pairs                            = 70
% -------------------------------------------------------------------------

%% ---------------- SETTINGS ----------------
dataFolder  = '/MATLAB Drive/Examples';% folder containing your sXX.mat files
numTrials   = 40;
fs          = 128;
numChannels = 32;              
filterOrder = 8;                
hfdKmax     = 6;                
subjFiles   = dir(fullfile(dataFolder, 's*.mat'));
numSubjects = numel(subjFiles);

assert(numSubjects > 0, 'No sXX.mat files found in %s', dataFolder);
fprintf('Found %d subject file(s) in %s\n', numSubjects, dataFolder);

channelNames = { ...
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1', ...
    'P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4', ...
    'F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8', ...
    'PO4','O2'};

pairNames    = {'Fp1_Fp2','AF3_AF4','F3_F4','F7_F8','FC5_FC6','FC1_FC2', ...
                 'C3_C4','T7_T8','CP5_CP6','CP1_CP2','P3_P4','P7_P8', ...
                 'PO3_PO4','O1_O2'};
leftPairIdx  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
rightPairIdx = [17,18,20,21,22,23,25,26,27,28,29,30,31,32];
numPairs     = numel(pairNames);

%% ---------------- FILTER DESIGN ----------------
bands = struct('delta',[0.5 4], 'theta',[4 8], 'alpha',[8 13], ...
                'beta',[13 30], 'gamma',[30 45]);
bandNames = fieldnames(bands);
numBands  = numel(bandNames);
bandFilters = cell(numBands,1);

for b = 1:numBands
    r = bands.(bandNames{b});
    bandFilters{b} = designfilt('bandpassiir', 'FilterOrder', filterOrder, ...
        'HalfPowerFrequency1', r(1), 'HalfPowerFrequency2', r(2), 'SampleRate', fs);
end
broadbandFilt = designfilt('bandpassiir', 'FilterOrder', filterOrder, ...
    'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 45, 'SampleRate', fs);

%% ---------------- BUILD FEATURE NAME LIST ----------------
DEnames = strings(1, numChannels*numBands);
RPnames = strings(1, numChannels*numBands);
idx = 1;
for b = 1:numBands
    for c = 1:numChannels
        DEnames(idx) = "DE_" + bandNames{b} + "_" + channelNames{c};
        RPnames(idx) = "RP_" + bandNames{b} + "_" + channelNames{c};
        idx = idx + 1;
    end
end

HFDnames = "HFD_" + string(channelNames);
hjLabels = ["Activity","Mobility","Complexity"];
HjNames  = strings(1, 3*numChannels);
idx = 1;
for f = 1:3
    for c = 1:numChannels
        HjNames(idx) = "Hjorth" + hjLabels(f) + "_" + channelNames{c};
        idx = idx + 1;
    end
end

AsymNames = strings(1, numPairs*numBands);
idx = 1;
for b = 1:numBands
    for p = 1:numPairs
        AsymNames(idx) = "DASM_" + bandNames{b} + "_" + pairNames{p};
        idx = idx + 1;
    end
end

allFeatNames = [DEnames, RPnames, HFDnames, HjNames, AsymNames];
numFeat = numel(allFeatNames);
fprintf('Total features per trial: %d\n', numFeat);

%% ---------------- PREALLOCATE ----------------
totalRows      = numSubjects * numTrials;
featureMatrix  = zeros(totalRows, numFeat);
subjectID      = zeros(totalRows,1);
trialID        = zeros(totalRows,1);
selfValence    = zeros(totalRows,1);
selfArousal    = zeros(totalRows,1);
row = 0;

%% ---------------- MAIN LOOP ----------------
for si = 1:numSubjects
    filePath = fullfile(dataFolder, subjFiles(si).name);
    subjNum  = str2double(regexp(subjFiles(si).name, '\d+', 'match', 'once'));
    S = load(filePath);
    fprintf('Processing subject %d/%d (%s)\n', si, numSubjects, subjFiles(si).name);
    
    for i = 1:numTrials
        DE  = zeros(numChannels, numBands);
        RP  = zeros(numChannels, numBands);
        HFDv = zeros(numChannels, 1);
        Hj  = zeros(numChannels, 3);
        
        for c = 1:numChannels
            sig = squeeze(S.data(i, c, :));
            sig = sig(385:end);   
            bb = filtfilt(broadbandFilt, sig);
            totalVar = var(bb) + eps;
            HFDv(c) = higuchiFD(bb, hfdKmax);
            
            d1 = diff(bb);
            d2 = diff(d1);
            act  = var(bb);
            mob  = sqrt(var(d1)/act);
            comp = sqrt(var(d2)/var(d1)) / mob;
            Hj(c,:) = [act, mob, comp];
            
            for b = 1:numBands
                bf = filtfilt(bandFilters{b}, sig);
                vb = var(bf);
                DE(c,b) = 0.5*log(2*pi*exp(1)*vb + eps);
                RP(c,b) = vb / totalVar;
            end
        end
        
        Asym = zeros(numPairs, numBands);
        for p = 1:numPairs
            Asym(p,:) = DE(rightPairIdx(p),:) - DE(leftPairIdx(p),:);
        end
        
        DErow = zeros(1, numChannels*numBands);
        RProw = zeros(1, numChannels*numBands);
        idx = 1;
        for b = 1:numBands
            for c = 1:numChannels
                DErow(idx) = DE(c,b);
                RProw(idx) = RP(c,b);
                idx = idx + 1;
            end
        end
        
        HFDrow = HFDv';
        Hjrow = zeros(1, 3*numChannels);
        idx = 1;
        for f = 1:3
            for c = 1:numChannels
                Hjrow(idx) = Hj(c,f);
                idx = idx + 1;
            end
        end
        
        Asymrow = zeros(1, numPairs*numBands);
        idx = 1;
        for b = 1:numBands
            for p = 1:numPairs
                Asymrow(idx) = Asym(p,b);
                idx = idx + 1;
            end
        end
        
        row = row + 1;
        featureMatrix(row,:) = [DErow, RProw, HFDrow, Hjrow, Asymrow];
        subjectID(row)   = subjNum;
        trialID(row)     = i;
        selfValence(row) = S.labels(i,1);
        selfArousal(row) = S.labels(i,2);
        
        if mod(i,10) == 0
            fprintf('  trial %d/%d\n', i, numTrials);
        end
    end
end

%% ---------------- ASSEMBLE AND SAVE ----------------
featTable = array2table(featureMatrix, 'VariableNames', cellstr(allFeatNames));
featTable.Subject        = subjectID;
featTable.Trial          = trialID;
featTable.SelfValence    = selfValence;
featTable.SelfArousal    = selfArousal;
featTable.ValenceLabel   = double(selfValence > 4.5);   % Adjusted to match reference paper
featTable.ArousalLabel   = double(selfArousal > 4.5);   % Adjusted to match reference paper

outPath = fullfile(dataFolder, 'eeg_features_all.csv');
writetable(featTable, outPath);
fprintf('\nSaved %d trials x %d features to %s\n', row, numFeat, outPath);

%% ---------------- LOCAL FUNCTION: HIGUCHI FRACTAL DIMENSION ----------------
function hfd = higuchiFD(x, kmax)
    x = x(:)';
    N = length(x);
    Lk = zeros(1, kmax);
    for k = 1:kmax
        Lmk = zeros(1, k);
        for m = 1:k
            idxSel = m:k:N;
            Xmk = x(idxSel);
            numPoints = length(Xmk);
            if numPoints < 2
                Lmk(m) = 0;
                continue;
            end
            Lm = sum(abs(diff(Xmk))) * (N - 1) / ((numPoints - 1) * k);
            Lmk(m) = Lm / k;
        end
        Lk(k) = mean(Lmk);
    end
    lnLk   = log(Lk + eps);
    lnInvK = log(1./(1:kmax));
    p = polyfit(lnInvK, lnLk, 1);
    hfd = p(1);
end


clc; clear all; close all;
%% EEG EMOTION CLASSIFICATION -- GRADIENT BOOSTING (RUSBoost)
% -------------------------------------------------------------------------
% Trains two SEPARATE binary classifiers (valence high/low, arousal
% high/low), each evaluated with subject-independent grouped
% cross-validation. Employs Random Under-Sampling Boosting (RUSBoost) 
% to correctly handle the imbalanced nature of the dataset.
% -------------------------------------------------------------------------

%% ---------------- SETTINGS ----------------
dataFolder = '/MATLAB Drive/Examples';
featureCSV = fullfile(dataFolder, 'eeg_features_all.csv');
K              = 8;     
numCycles      = 150;   
learnRate      = 0.1;
weakLearnDepth = 20;    
topNFeatures   = 20;    
rng(42);   

%% ---------------- LOAD FEATURES ----------------
%assert(isfile(featureCSV), 'Feature file not found: %s\nRun EEG_Feature_Extraction_FullSet.m first.', featureCSV);
T = readtable(featureCSV);
fprintf('Loaded %d trials with %d columns\n', height(T), width(T));

metaCols = {'Subject','Trial','SelfValence','SelfArousal','ValenceLabel','ArousalLabel'};
featureCols = setdiff(T.Properties.VariableNames, metaCols, 'stable');
fprintf('Using %d features\n', numel(featureCols));
X = table2array(T(:, featureCols));
subjectID = T.Subject;

%% ---------------- PER-SUBJECT Z-SCORING ----------------
subjectsList = unique(subjectID);
Xz = zeros(size(X));
for su = 1:numel(subjectsList)
    idx = subjectID == subjectsList(su);
    mu = mean(X(idx,:), 1);
    sigma = std(X(idx,:), 0, 1);
    sigma(sigma == 0) = 1;   
    Xz(idx,:) = (X(idx,:) - mu) ./ sigma;
end

%% ---------------- SUBJECT-GROUPED K-FOLD ASSIGNMENT ----------------
shuffledSubjects = subjectsList(randperm(numel(subjectsList)));
foldOfSubject = mod((0:numel(shuffledSubjects)-1), K) + 1;
subjectFoldMap = containers.Map(num2cell(shuffledSubjects), num2cell(foldOfSubject));

foldOfTrial = zeros(height(T),1);
for r = 1:height(T)
    foldOfTrial(r) = subjectFoldMap(subjectID(r));
end

%% ---------------- TRAIN + EVALUATE ----------------
targets = {'ValenceLabel','ArousalLabel'};
results = struct();

for t = 1:numel(targets)
    targetName = targets{t};
    Y = T.(targetName);
    
    fprintf('\n=== Target: %s ===\n', targetName);
    
    foldAcc = zeros(K,1);
    allPred = zeros(height(T),1);
    allTrue = Y;
    
    for k = 1:K
        testIdx  = (foldOfTrial == k);
        trainIdx = ~testIdx;
        
        t0 = templateTree('MaxNumSplits', weakLearnDepth);
        Mdl = fitcensemble(Xz(trainIdx,:), Y(trainIdx), ...
            'Method', 'RUSBoost', ...
            'NumLearningCycles', numCycles, ...
            'Learners', t0, ...
            'LearnRate', learnRate, ...
            'ClassNames', [0 1]);
            
        pred = predict(Mdl, Xz(testIdx,:));
        allPred(testIdx) = pred;
        
        foldAcc(k) = mean(pred == Y(testIdx)) * 100;
        fprintf('  Fold %d/%d (%d test trials): %.1f%% accuracy\n', ...
            k, K, sum(testIdx), foldAcc(k));
    end
    
    overallAcc = mean(allPred == allTrue) * 100;
    cm = confusionmat(allTrue, allPred);
    fprintf('\n%s -- mean fold accuracy: %.1f%% (+/- %.1f%%), pooled accuracy: %.1f%%\n', ...
        targetName, mean(foldAcc), std(foldAcc), overallAcc);
    fprintf('Confusion matrix (rows=true, cols=predicted), classes [0 1]:\n');
    disp(cm);
    
    % ---- Final model on ALL data ----
    t0 = templateTree('MaxNumSplits', weakLearnDepth);
    finalMdl = fitcensemble(Xz, Y, 'Method', 'RUSBoost', ...
        'NumLearningCycles', numCycles, 'Learners', t0, ...
        'LearnRate', learnRate, 'ClassNames', [0 1]);
        
    imp = predictorImportance(finalMdl);
    [sortedImp, sortIdx] = sort(imp, 'descend');
    topFeatNames = featureCols(sortIdx(1:topNFeatures));
    
    fprintf('\nTop %d most important features for %s:\n', topNFeatures, targetName);
    for f = 1:topNFeatures
        fprintf('  %2d. %-30s %.4f\n', f, topFeatNames{f}, sortedImp(f));
    end
    
    results.(targetName).foldAcc = foldAcc;
    results.(targetName).overallAcc = overallAcc;
    results.(targetName).confusionMatrix = cm;
    results.(targetName).model = finalMdl;
    results.(targetName).topFeatures = table(topFeatNames', sortedImp(1:topNFeatures)', ...
        'VariableNames', {'Feature','Importance'});
        
    writetable(results.(targetName).topFeatures, ...
        fullfile(dataFolder, sprintf('%s_top_features.csv', targetName)));
end


%% ---------------- SUMMARY ----------------
fprintf('\n========== SUMMARY ==========\n');
fprintf('Valence : mean fold accuracy %.1f%%, pooled accuracy %.1f%%\n', ...
    mean(results.ValenceLabel.foldAcc), results.ValenceLabel.overallAcc);
fprintf('Arousal : mean fold accuracy %.1f%%, pooled accuracy %.1f%%\n', ...
    mean(results.ArousalLabel.foldAcc), results.ArousalLabel.overallAcc);
    
save(fullfile(dataFolder, 'eeg_xgboost_equivalent_results.mat'), 'results');
fprintf('\nSaved trained models and results to eeg_xgboost_equivalent_results.mat\n');



clc; clear all; close all;
%% PARSIMONY CHECK -- full 518-feature model vs top-N most important features
% -------------------------------------------------------------------------
% OPTIMIZED VERSION (RUSBoost): Fixes severe class imbalance by using 
% Random Under-Sampling Boosting. In every boosting round, the majority 
% class is downsampled so the tree sees a balanced split, forcing it 
% to learn the minority class.
% -------------------------------------------------------------------------

%% ---------------- SETTINGS ----------------
dataFolder = '/MATLAB Drive/Examples';
featureCSV = fullfile(dataFolder, 'eeg_features_all.csv');
K              = 8;
numCycles      = 150;
learnRate      = 0.1;
weakLearnDepth = 20;
topN           = 25;
rng(42);

%% ---------------- LOAD ----------------
%assert(isfile(featureCSV), 'Feature file not found: %s\nRun EEG_Feature_Extraction_FullSet.m first.', featureCSV);
T = readtable(featureCSV);
fprintf('Loaded %d trials with %d columns\n', height(T), width(T));

metaCols = {'Subject','Trial','SelfValence','SelfArousal','ValenceLabel','ArousalLabel'};
featureCols = setdiff(T.Properties.VariableNames, metaCols, 'stable');
X = table2array(T(:, featureCols));
subjectID = T.Subject;

%% ---------------- CLASS BALANCE CHECK ----------------
fprintf('\n--- Class balance check ---\n');
for tName = {'ValenceLabel','ArousalLabel'}
    Yc = T.(tName{1});
    n0 = sum(Yc==0); n1 = sum(Yc==1);
    fprintf('%s: Low=%d (%.1f%%), High=%d (%.1f%%)  [majority-class baseline = %.1f%%]\n', ...
        tName{1}, n0, 100*n0/height(T), n1, 100*n1/height(T), 100*max(n0,n1)/height(T));
end

%% ---------------- PER-SUBJECT Z-SCORING ----------------
subjectsList = unique(subjectID);
Xz = zeros(size(X));
for su = 1:numel(subjectsList)
    idx = subjectID == subjectsList(su);
    mu = mean(X(idx,:), 1);
    sigma = std(X(idx,:), 0, 1);
    sigma(sigma == 0) = 1;
    Xz(idx,:) = (X(idx,:) - mu) ./ sigma;
end

%% ---------------- SUBJECT-GROUPED K-FOLD ASSIGNMENT ----------------
shuffledSubjects = subjectsList(randperm(numel(subjectsList)));
foldOfSubject = mod((0:numel(shuffledSubjects)-1), K) + 1;
subjectFoldMap = containers.Map(num2cell(shuffledSubjects), num2cell(foldOfSubject));

foldOfTrial = zeros(height(T),1);
for r = 1:height(T)
    foldOfTrial(r) = subjectFoldMap(subjectID(r));
end

%% ---------------- TARGET LOOP ----------------
targets = {'ValenceLabel','ArousalLabel'};
parsimonyResults = struct();

for t = 1:numel(targets)
    targetName = targets{t};
    Y = T.(targetName);
    
    fprintf('\n=== Target: %s ===\n', targetName);
    
    % ---- Full feature set ----
    fprintf('-- Full feature set (%d features) --\n', size(Xz,2));
    [foldAccFull, overallAccFull, cmFull, balAccFull, f1MacroFull] = ...
        runCV(Xz, Y, foldOfTrial, K, numCycles, learnRate, weakLearnDepth);
        
    fprintf('Full set: accuracy %.1f%% | balanced accuracy %.1f%% | macro-F1 %.2f\n', ...
        overallAccFull, balAccFull, f1MacroFull);
        
    % ---- Feature importance from a RUSBoost model trained on all data ----
    t0 = templateTree('MaxNumSplits', weakLearnDepth);
    fullMdl = fitcensemble(Xz, Y, 'Method','RUSBoost', 'NumLearningCycles', numCycles, ...
        'Learners', t0, 'LearnRate', learnRate, 'ClassNames', [0 1]); 
        
    imp = predictorImportance(fullMdl);
    [~, sortIdx] = sort(imp, 'descend');
    topIdx = sortIdx(1:topN);
    topNames = featureCols(topIdx);
    
    % ---- Pruned feature set ----
    Xtop = Xz(:, topIdx);
    fprintf('\n-- Top %d features only --\n', topN);
    
    [foldAccTop, overallAccTop, cmTop, balAccTop, f1MacroTop] = ...
        runCV(Xtop, Y, foldOfTrial, K, numCycles, learnRate, weakLearnDepth);
        
    fprintf('Top %d: accuracy %.1f%% | balanced accuracy %.1f%% | macro-F1 %.2f\n', ...
        topN, overallAccTop, balAccTop, f1MacroTop);
        
    % ---- Store Results ----
    parsimonyResults.(targetName).fullAcc         = overallAccFull;
    parsimonyResults.(targetName).topAcc          = overallAccTop;
    parsimonyResults.(targetName).balAccFull      = balAccFull;
    parsimonyResults.(targetName).balAccTop       = balAccTop;
    parsimonyResults.(targetName).f1MacroFull     = f1MacroFull;
    parsimonyResults.(targetName).f1MacroTop      = f1MacroTop;
    parsimonyResults.(targetName).foldAccFull     = foldAccFull;
    parsimonyResults.(targetName).foldAccTop      = foldAccTop;
    parsimonyResults.(targetName).topFeatureNames = topNames;
    parsimonyResults.(targetName).cmFull          = cmFull;
    parsimonyResults.(targetName).cmTop           = cmTop;
    
    outT = table(topNames', 'VariableNames', {'Feature'});
    writetable(outT, fullfile(dataFolder, sprintf('%s_top%d_features_used.csv', targetName, topN)));
end

%% ---------------- SUMMARY ----------------
fprintf('\n========== PARSIMONY CHECK SUMMARY (RUSBoost) ==========\n');
fprintf('Valence : full acc %.1f%% (bal %.1f%%, F1m %.2f)  vs  top%d acc %.1f%% (bal %.1f%%, F1m %.2f)\n', ...
    parsimonyResults.ValenceLabel.fullAcc, parsimonyResults.ValenceLabel.balAccFull, parsimonyResults.ValenceLabel.f1MacroFull, ...
    topN, parsimonyResults.ValenceLabel.topAcc, parsimonyResults.ValenceLabel.balAccTop, parsimonyResults.ValenceLabel.f1MacroTop);
    
fprintf('Arousal : full acc %.1f%% (bal %.1f%%, F1m %.2f)  vs  top%d acc %.1f%% (bal %.1f%%, F1m %.2f)\n', ...
    parsimonyResults.ArousalLabel.fullAcc, parsimonyResults.ArousalLabel.balAccFull, parsimonyResults.ArousalLabel.f1MacroFull, ...
    topN, parsimonyResults.ArousalLabel.topAcc, parsimonyResults.ArousalLabel.balAccTop, parsimonyResults.ArousalLabel.f1MacroTop);
    
save(fullfile(dataFolder, 'eeg_parsimony_check_results.mat'), 'parsimonyResults');
fprintf('\nSaved to eeg_parsimony_check_results.mat\n');

%% ---------------- LOCAL FUNCTION ----------------
function [foldAcc, overallAcc, cm, balAcc, f1Macro] = runCV(Xdata, Y, foldOfTrial, K, numCycles, learnRate, weakLearnDepth)
% Subject-grouped K-fold CV using RUSBoost to explicitly handle class imbalance
% by randomly under-sampling the majority class in every boosting iteration.
    
    foldAcc = zeros(K,1);
    allPred = zeros(size(Y));
    
    for k = 1:K
        testIdx  = (foldOfTrial == k);
        trainIdx = ~testIdx;
        
        Ytrain = Y(trainIdx);
        
        t0 = templateTree('MaxNumSplits', weakLearnDepth);
        Mdl = fitcensemble(Xdata(trainIdx,:), Ytrain, 'Method','RUSBoost', ...
            'NumLearningCycles', numCycles, 'Learners', t0, ...
            'LearnRate', learnRate, 'ClassNames', [0 1]); 
            
        pred = predict(Mdl, Xdata(testIdx,:));
        allPred(testIdx) = pred;
        
        foldAcc(k) = mean(pred == Y(testIdx)) * 100;
    end
    
    overallAcc = mean(allPred == Y) * 100;
    cm = confusionmat(Y, allPred);
    
    rec0 = cm(1,1)/(cm(1,1)+cm(1,2)+eps);
    rec1 = cm(2,2)/(cm(2,2)+cm(2,1)+eps);
    balAcc = 100*mean([rec0, rec1]);
    
    prec0 = cm(1,1)/(cm(1,1)+cm(2,1)+eps);
    prec1 = cm(2,2)/(cm(2,2)+cm(1,2)+eps);
    
    f1_0 = 2*prec0*rec0/(prec0+rec0+eps);
    f1_1 = 2*prec1*rec1/(prec1+rec1+eps);
    f1Macro = mean([f1_0, f1_1]);
end
