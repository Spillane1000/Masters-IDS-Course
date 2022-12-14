% input format: 
% --for input matrix simply a matrix of all data matlab will automatically
% convert all values to strings whether numerical or not. 
% Example -- input_matrix = ["X","A",6;"Y","B",NaN;NaN,NaN,NaN;"X","C",8;"Y","D",12];
% inputs of type table also accepted however missing values must be
% represented as <missing> and not null entries for function to implement
% imputation correctly

% -- for s an array stating whether the corresponding column in the input
% matrix is continuous or categorical. "Cont" for Continuous and "Cat" for
% Categorical. incorrect input will output the column without data imputed
% Example -- s = ["Cat","Cat","Cont"];

% Note: for the mode of categorical data, if there are two or more values
% which occur the most, Matlab will automatically choose the smallest then
% first alphabetical value 


function X_full = myImpute(input_matrix,s)

% if the input type is of class 'table' conver to matrix of strings for use
% in the function
input_matrix2 = strings(0);
if istable(input_matrix)
    for i = 1:width(input_matrix)
        input_matrix2(:,i) = string(table2array(input_matrix(:,i)));
    end
else
    input_matrix2 = input_matrix;
end


    X_full = strings(size(input_matrix2)); % store for final fully imputed data
     for f = 1:length(s)   
         % for when dealing with Continuous variable
        if contains(s(f),"Cont") 
                % Converts String data back to numerical as Matlab
                % automatically converts the continuous data to strings
                conv_to_num = [];                                         
                for j = 1:length(input_matrix2(:,f))                       
                    if ismissing(input_matrix2(j,f))                       
                        conv_to_num(j) = NaN;                              
                    else                                                   
                        conv_to_num(j) = str2num(input_matrix2(j,f));     
                    end                                                   
                end
                imputed_array = []; % array with imputed data i.e. no missing
                for k = 1:length(conv_to_num)
                    if isnan(conv_to_num(k))
                        imputed_array(k) = mean(conv_to_num,'omitnan'); % imputes mean where data is missing
                    else
                        imputed_array(k) = conv_to_num(k); % number stays the same if not missing
                    end
                end
        % fro when dealing with categorical data               
        elseif contains(s(f),"Cat")
                imputed_array = strings(0) ; % store for data after imputation
                to_impute = input_matrix2(:,f); % data where imputation is to be done 
                for p = 1:length(to_impute)
                    if ismissing(to_impute(p))
                        imputed_array(p) = string(mode(categorical(to_impute))); % convert to categorical variables computes mode and converts back to string
                    else
                        imputed_array(p) = to_impute(p);
                    end
                end
        else
            fprintf("Error: variable type not specified correctly please use Cat for Categorical and Cont for Continuous \nWARNING!!! imputation not fully carried out")
            imputed_array = input_matrix2(:,f);
        end
        X_full(:,f) = imputed_array; % stores the imputed data in relevant column
     end
end



