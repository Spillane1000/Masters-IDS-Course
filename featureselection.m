function featureselection()
    nrepeats = 20 ;

    data = csvread('diabetes.csv');

    col_name = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'};
    X = data(:,1:8);
    y = data(:,9); % 1: has diabetes, 0: no diabetes

    [n, d] = size(X);
    
    %% ******* training a classifier on all features **********
    % SVM
    mdl = fitcsvm(X, y);
    [ypred, ~] = predict(mdl, X);
    acc = mean(ypred==y) ;

    %% ******* performing feature shuffling
    acc_perm = zeros(nrepeats, d);
    for i=1:nrepeats
        acc_perm(i, :) = featureshuffling(X, y, mdl);
    end
    importances = acc - mean(acc_perm, 1);

    figure,
    errorbar(importances, std(acc_perm, 1), 's', ...
            'MarkerSize',3, 'Color', 'k', ...
            'LineWidth', 1.5, 'CapSize', 8)
    set(gca, 'XTick', 1:8, 'XTickLabel',col_name)
    xlim([0.5, 8.5])
end

function acc = featureshuffling(X, y, mdl)
    [n, d] = size(X);
    acc = size(1,d);
    
    %% ********** Here you can add your code **************
    perm = repmat(X, 1, 1, d); % create d copies of X
    for i = 1:d                % permute the ith column of the ith copy of X, leaving the other columns intact
        perm(:,i,i) = X(randperm(n),i);
    end
    for j = 1:d                
        [ypred, ~] = predict(mdl, perm(:,:,j)); % test the classifier against X with the jth column shuffled
        accuracy = mean(ypred==y);              % calculate the accuracy
        acc(j)=accuracy ;                       % fill acc with the values of the accuracy for each feature
    end
end


