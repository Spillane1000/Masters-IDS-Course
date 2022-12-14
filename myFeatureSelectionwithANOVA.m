% Application of face recognition to demonstrate the effectiveness of ANOVA in
% feature selection
% Course: Introduction to Data Science
% Author: George Azzopardi - September 2018

function myFeatureSelectionwithANOVA

% directory names of image data
trainingFolder = 'data/training-set/';
testingFolder = 'data/testing-set/';

% Convert training and test data into column vectors
[trainingLabel, trainingData, trainingimgs] = getData(trainingFolder);
[testLabel, testData, testingimgs] = getData(testingFolder);

labelList = [trainingLabel,testLabel];
labelId = grp2idx(labelList);

trainingId = labelId(1:numel(trainingLabel));
testId = labelId(numel(trainingLabel)+1:end);

for i = 1:size(trainingData,1)
    F(i) = myOneWayANOVA(trainingData(i,:), trainingId);
end

[~,srtidx] = sort(F,'descend');

acc = zeros(1,size(trainingData,1));
for i = 1:size(trainingData,1)
    knn = fitcknn(trainingData(srtidx(1:i),:)',trainingId);
    c1 = predict(knn,testData(srtidx(1:i),:)');            
    acc(i) = sum(testId == c1)/numel(testId);
end

figure;plot(acc);
xlabel('Number of Selected Features');
ylabel('Accuracy Rate');

function [labelList, dataMatrix, im] = getData(inputFolder)
d = dir([inputFolder,'*.jpg']);
dataMatrix = [];
filenameList = cell(0);
for i = 1:numel(d)
    filenameList{i} = [inputFolder,d(i).name];
    im{i} = imread(filenameList{i});
    dataMatrix(:,i) = im{i}(:);    
    [~,filename,~] = fileparts(filenameList{i});    
    labelList{i} = filename(1:4);
end