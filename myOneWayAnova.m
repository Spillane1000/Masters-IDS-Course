% IV is an array of values, DV indicates the classes that the values fall into
function F = myOneWayAnova(IV, DV)
    % Input checking
    if (length(unique(DV)) == 1)
        error("DV only contains one unique value!");
    end
    if (length(IV) ~= length(DV))
        error("IV and DV must have the same number of elements!");
    end
        numItems = length(IV);
    
    % Create a mapping from categories to their scores (e.g., 'H' -> [4,5,6])
    categories = containers.Map
    for i=1:numItems
        if (isKey(categories, DV(i)))
            categories(DV(i)) = [categories(DV(i)), IV(i)];
        else
            categories(DV(i)) = [IV(i)];
        end
    end
    
    % Mapping of sums of squares between groups
    mapSSB = containers.Map
    k = keys(categories)
    val = values(categories)
    for i=1:length(categories)
        mapSSB(k{i}) = length(val{i})*(mean(val{i})-mean(IV))^2
    end
    
    % Calculate mean sum of squares between groups
    SSB = (1/(length(k)-1))*sum(cell2mat(values(mapSSB)));
    
    % Mapping of sums of squares within groups
    mapSSW = containers.Map
    dfSum = 0
    for i=1:length(categories)
        mapSSW(k{i}) = (length(val{i})-1)*var(val{i})
        dfSum = dfSum + length(val{i})-1
    end
    
    % Calculate sums of squares within groups
    SSW = sum(cell2mat(values(mapSSW)))/dfSum
    
    % Calculate and round F-score
    F = SSB/SSW
    F = round(F, 3)
end
