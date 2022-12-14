% Input A is an N by M matrix.
% Returns a list of pairs (tuples) each containing an L2-normalized
% principal component (as a NumPy array)
% and its corresponding eigenvalue, sorted in descending order by eigenvalue.
function [pc,eigenvalues] = mypca(A)
    % Center the data
    meanDataPoint = mean(A, 1);
    CenteredData = A - meanDataPoint;
    
    % Calculate the covariance matrix
    CovMatrix = cov(CenteredData);
    
    % EigenVals is a matrix with the eigen values on the diagonal
    % The rows (eigenvectors) in EigenVecs are readily L2-normalized
    [EigenVecs,EigenVals] = eig(CovMatrix);
    
    % Mold values into desired output format, sorting by eigenvalues in
    % descending order
    eigenvalues = diag(EigenVals);
    [~,srtidx] = sort(eigenvalues, 'descend');
    eigenvalues = eigenvalues(srtidx);
    pc = EigenVecs(:,srtidx);
end
