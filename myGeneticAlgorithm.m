% Genetic algorithm for the selection of the best subset of features
% Course: Introduction to Data Science
% Author: George Azzopardi
% Date:   October 2019

function bestchromosome = myGeneticAlgorithm(features,labels)
    % features = A matrix of independent variables
    % labels = A vector that contains the labels for each rows in matrix features

    nchroms       = 100; % number of chromosomes
    nepochs       = 10;  % number of epochs
    nparentsratio = 0.2; % portion of elite list
    mutateprob    = 0.1; % probability to mutate a bit in a chromosome

    % Create figure that shows the progress of the genetic algorithm
    figure;hold on;
    title('Feature Selection with Genetic Algorithm');
    colorlist = jet(nepochs);

    % Convert labels, which can be in string format, to numeric.
    % E.g., labels = ["hi", "bye", "bye", "bye"]
    %       lbls   = [1   ; 2    ; 2    ; 2    ]
    [lbls,~] = grp2idx(labels);

    % Iterate through all epochs
    for epoch = 1:nepochs
        fprintf('epoch %d of %d\n',epoch,nepochs);
        if epoch == 1
            % generate the intial popultion of chromosome with randomly
            % assigned bits
            nFeatures = size(features, 2);
            pop = generateInitialPopulation(nchroms, nFeatures);        
        else
            % generate a new population by creating offspring from the best
            % performing chromosome (or parents)
            pop = getnewpopulation(pop,score,nparentsratio,mutateprob);
        end
        pop = logical(pop);

        % Compute the fitness score for each chromosome
        score = zeros(1,nchroms);
        for i = 1:nchroms
            score(i) = getScore(pop(i,:),features,lbls);    
        end

        % Plot the scores to visualize the progress
        plot(sort(score,'descend'),'color',colorlist(epoch,:));
        xlabel('Chromosome');
        ylabel('Fitness Score');
        legendList{epoch} = sprintf('Epoch %d',epoch);
        legend(legendList);
        drawnow;
    end

    % Return the chromosome with the maximum fitness score
    [~,mxind] = max(score);
    bestchromosome = pop(mxind,:);

function newpop = getnewpopulation(pop,score,nparentsratio,mutateprob)
    % Generate a new population by first selecting the best performing
    % chromosomes from the given pop matix, and subsequently generate new offspring chromosomes from randomly
    % selected pairs of parent chromosomes.

    % Step 1. Write code to select the top performing chromosomes. Use nparentsratio to
    % calculate how many parents you need. If pop has 100 rows and
    % nparentsration is 0.2, then you have to select the top performing 20
    % chromosomes
    nparents = round(size(pop, 1)*nparentsratio);
    [~, maxIndices] = maxk(score, nparents);
    
    newpop = pop(maxIndices,:);

    % Step 2. Iterate until a new population of the same size is generated. 
    % In each iteration create a new offspring chromosome from two randomly selected 
    % parent chromosomes. Use the function getOffSpring to generate a new offspring.
    for i=nparents+1:100
        parents = newpop(randperm(nparents,2),:);
        newpop(i,:) = getOffSpring(parents(1,:),parents(2,:),mutateprob);
    end

function offspring = getOffSpring(parent1,parent2,mutateprob)
    % Generate an offspring from parent1 and parent2 and mutate the bits by
    % using the probability mutateprob.
    [~, length] = size(parent1);

    offspring = zeros(1, length);
    cutOff = randsample(length-1, 1);
    offspring(1:cutOff) = parent1(1:cutOff);
    offspring(cutOff+1:length) = parent2(cutOff+1:length);

    % Step 2. Write code to mutate some bits with given mutation 
    % probability mutateprob
    for i=1:length
        if rand() < mutateprob
            offspring(i) = ~offspring(i);
        end
    end
    
    % If all offspring bits are 0, the score function fails (and feature 
    % selection will be impossible with 0 features), so assign a 1 randomly
    if all(offspring==0)
        offspring(randsample(length, 1)) = 1;
    end

function score = getScore(chromosome,train_feats,labels)
    % Compute the fitness score using 2-fold cross validation and KNN
    % classifier

    cv = cvpartition(labels,'Kfold',2);
    for i = 1:cv.NumTestSets
        knn = fitcknn(train_feats(cv.training(i),chromosome),labels(cv.training(i)));
        c = predict(knn,train_feats(cv.test(i),chromosome));
        acc(i) = sum(c == labels(cv.test(i)))/numel(c);
    end
    meanacc = mean(acc);
    score = (10^4 * meanacc) + (0.4 * sum(chromosome == 0));

function pop = generateInitialPopulation(n,ndim)
    % Generate the initial population of chromosomes with random bits
    % Addition: random bits meaning random in which bit (although the first 
    % row is all 1s), not in number
    % E.g., with 5-bit chromosomes, the number of 1s per row repeats the
    % pattern 3,1,2,3,1,2,3,1,2,3,..

    pop = zeros(n,ndim);

    pop(1,:) = ones(1,ndim);
    for i = 2:n    
        pop(i,randperm(ndim,mod(i,round(ndim/2))+1)) = 1;
    end
