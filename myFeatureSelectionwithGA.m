% Application of face recognition to demonstrate the effectiveness of GA in
% feature selection
% Course: Introduction to Data Science
% Author: George Azzopardi - September 2018

function myFeatureSelectionwithGA

% directory names of image data
trainingFolder = 'data/training-set/';
testingFolder = 'data/testing-set/';

% Convert training and test data into column vectors
[trainingLabel, trainingData, trainingimgs] = getData(trainingFolder);
[testLabel, testData, testingimgs] = getData(testingFolder);

meanTrainingFace = mean(trainingData,2);
trainingData = trainingData - repmat(meanTrainingFace,1,size(trainingData,2));
testData = testData - repmat(meanTrainingFace,1,size(testData,2));

labelList = [trainingLabel,testLabel];
labelId = grp2idx(labelList);

trainingId = labelId(1:numel(trainingLabel));
testId = labelId(numel(trainingLabel)+1:end);

bestchromosome = myGeneticAlgorithm(trainingData',trainingId);

knn = fitcknn(trainingData(bestchromosome,:)',trainingId);
c1 = predict(knn,testData(bestchromosome,:)');
acc1 = sum(c1 == testId)/numel(c1);    

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