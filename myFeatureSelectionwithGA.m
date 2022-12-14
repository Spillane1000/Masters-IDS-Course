% Application of face recognition to demonstrate the effectiveness of GA in
% feature selection
% Course: Introduction to Data Science
% Author: George Azzopardi - October 2019

function [bestchromosome, acc1, acc2] = myFeatureSelectionwithGA
    % Note the change in function signature to return acc1 and acc2. This
    % is for the sake of plotting them against each other with
    % featureSelectionPlots.
    % load data
    load wine.data;
    labels = wine(:,1);
    features = wine(:,2:end);

    % Split data into training (70%) and test (30%) sets
    c = cvpartition(labels,'holdout', 0.3,'Stratify',true);
    trainingData = features(c.training,:);
    trainingLabel = labels(c.training);
    testData = features(c.test,:);
    testLabel = labels(c.test);

    % Retrieve the best feature set using GA on the training data
    bestchromosome = myGeneticAlgorithm(trainingData,trainingLabel);
    
    % trainingData(:,bestchromosome) is the trainingData with the least
    % important features removed.
    knn = fitcknn(trainingData(:,bestchromosome),trainingLabel);
    c1 = predict(knn,testData(:,bestchromosome));
    acc1 = sum(c1 == testLabel)/numel(c1);
    fprintf('Feature size: %d\n', sum(bestchromosome)); 
    fprintf('accuracy: %2.6f\n',acc1); 

    knn = fitcknn(trainingData,trainingLabel);
    c1 = predict(knn,testData);
    acc2 = sum(c1 == testLabel)/numel(c1);
    fprintf('All features\n'); 
    fprintf('accuracy: %2.6f\n',acc2);