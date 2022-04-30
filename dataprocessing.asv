% import data
clear
%%
[pnotes,train] = datasorting;
features = readtable('features.csv');
train = cell2table(train);
%% preprocessing
clc
tic
pnhist = pnotes(2:end,3); 
tokens = tokenizedDocument;

oldNgrams = ["yo"
            "m"
            "f"
            "mo"
            'mi'];
newNgrams = ["year" "old"
             "male" ""
             "female" ""
             "month" ""
             "myocardial" "infarction"];

for r = 1:size(pnhist,1)
    t = erasePunctuation(pnhist{r});
    t = tokenizedDocument(sprintf('%s',t));
    t = removeStopWords(t);
    t = normalizeWords(t);
    t = replaceNgrams(t,oldNgrams,newNgrams);
    t = correctSpelling(t);
    tokens = [tokens;t];
    fprintf('Iteration: %d\n',r);
end
toc
%% moeldingnng
mdl = bert;
tokenizer = mdl.Tokenizer;

train.Tokens = encode(tokenizer, train.train5);

train.train4 = categorical(train.train4);
classes = categories(train.train4);
nClasses = numel(classes);

cvp = cvpartition(classes,"Holdout",0.2);
dataTrain = train(training(cvp),:);
dataTest = train(test(cvp),:);

nTrain = size(dataTrain,1)
nTest = size(dataTest,1)

%%
textTrain = dataTrain.train5;
testTest = dataTest.train5;

TTrain = dataTrain.train4;
TTest = dataTest.train4;

tokensTrain = dataTrain.Tokens;
tokensTest = dataTest.Tokens;

figure
wordcloud(textTrain);
title("Training Data")

%%
dsXTrain = arrayDatastore(tokensTrain,"OutputType","same");
dsTTrain = arrayDatastore(TTrain);
cdsTrain = combine(dsXTrain,dsTTrain);

dsXTest = arrayDatastore(tokensTest,"OutputType","same");
dsTTest = arrayDatastore(TTest);
cdsTest = combine(dsXTest,dsTTest);

miniBatchSize = 64;
paddingValue = mdl.Tokenizer.PaddingCode;
maxSequenceLength = mdl.Parameters.Hyperparameters.NumContext;

mbqTrain = minibatchqueue(cdsTrain,1,"MiniBatchSize",miniBatchSize,"MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));
mbqTest = minibatchqueue(cdsTest,1,"MiniBatchSize",miniBatchSize,"MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

if canUseGPU
    mdl.Parameters.Weights = dlupdate(@gpuArray,mdl.Parameters.Weights);
end

%%

extractedFeatures_train = [];
reset(mbqTrain);
while hasdata(mbqTrain)
    X = next(mbqTrain);
    f = bertEmbed(X,mdl.Parameters);
    extractedFeatures_train = [extractedFeatures_train gather(extractdata(f))];
end
extractedFeatures_train = extractedFeatures_train.';

extractedFeatures_test = [];
reset(mbqTest);
while hasdata(mbqTest)
    X = next(mbqTest);
    features = bertEmbed(X,mdl.Parameters);
    extractedFeatures_test = cat(2,extractedFeatures_test,gather(extractdata(features)));
end
extractedFeatures_test = extractedFeatures_test.';

%%

nFeatures = mdl.Parameters.Hyperparameters.HiddenSize;
layers = [
    featureInputLayer(nFeatures)
    fullyConnectedLayer(nClasses)
    softmaxLayer
    classificationLayer];

opts = trainingOptions('adam',...
    "MiniBatchSize",64,...
    "ValidationData",{extractedFeatures_test,dataTest.train4},...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",0,MaxEpochs=800);

net = trainNetwork(extractedFeatures_train,dataTrain.train4,layers,opts);

YPredValidation = classify(net,extractedFeatures_test);

figure
confusionchart(TTest,YPredValidation)

accuracy = mean(dataTest.train4 == YPredValidation)

%% Try new data
newData_raw = readcell('patient_notes.csv');
featuresList = readcell('features.csv');
featuresList(1,:)=[];

rIdx = randi(length(featuresList));
newData = convertCharsToStrings(newData_raw{rIdx,3})
tokensNew = encode(tokenizer,newData);
tokensNew = padsequences(tokensNew,2,"PaddingValue",tokenizer.PaddingCode);

featuresNew = bertEmbed(tokensNew,mdl.Parameters)';
featuresNew = gather(extractdata(featuresNew));
labelsNew = classify(net,featuresNew);

idx = find(cell2mat(featuresList(:,1))== double(labelsNew));
extractedFeat = featuresList{idx,3}





% Supporting Functions
% Credit to bwdGitHub on Github

%%% Predictors Preprocessing Functions
% The |preprocessPredictors| function truncates the mini-batches to have
% the specified maximum sequence length, pads the sequences to have the
% same length. Use this preprocessing function to preprocess the predictors
% only.
function X = preprocessPredictors(X,paddingValue,maxSeqLen)

X = truncateSequences(X,maxSeqLen);
X = padsequences(X,2,"PaddingValue",paddingValue);

end

%%% BERT Embedding Function
% The |bertEmbed| function maps input data to embedding vectors and
% optionally applies dropout using the "DropoutProbability" name-value
% pair.
function Y = bertEmbed(X,parameters,args)

arguments
    X
    parameters
    args.DropoutProbability = 0
end

dropoutProbabilitiy = args.DropoutProbability;

Y = bert.model(X,parameters, ...
    "DropoutProb",dropoutProbabilitiy, ...
    "AttentionDropoutProb",dropoutProbabilitiy);

% To return single feature vectors, return the first element.
Y = Y(:,1,:);
Y = squeeze(Y);

end
