% Classical MDS code

numPoints = 1000;
numDimensions = 2;
path = imread('lena.jpg');
% imshow(path)

% Get user input
userChoice = input('Enter your choice: ', 's');  % 's' indicates input as string

% Check user input and execute corresponding function
if strcmp(userChoice, 'data1')
    % Execute function1
    X = swissRole(numPoints);
elseif strcmp(userChoice, 'data2')
    % Execute function2
    X = imageLoad(path);
end

[embedding, stress] = classicalMDS(X, numDimensions);
Y = tsne(X, 'Algorithm','exact','NumDimensions',3);
Y = tsne(X2);
%disp(X);
%disp(embedding);
%disp(Y);
fig = figure;
scatter(Y(:, 1), Y(:, 2), 'filled');
colormap('jet');
colorbar;
xlabel('Dimension 1');
ylabel('Dimension 2');
title('tSNE Embedding');

% Func to generate the swissRole dataset
function X = swissRole(numPoints)

    % Generate Swiss roll data
    % numPoints - Number of points in the Swiss roll
    t = (3 * pi / 2) * (1 + 2 * rand(numPoints, 1));   % Generate the t parameter
    height = 21 * rand(numPoints, 1);                   % Generate the height parameter
    X = [(t .* cos(t)), height, (t .* sin(t))];         % Create the Swiss roll data
    
    % Plot the Swiss roll
    figure;
    scatter3(X(:,1), X(:,2), X(:,3), 20, X(:,3), 'filled');
    colormap('jet');
    colorbar;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Swiss Roll');

end

% Func to load the image data
function imageVector = imageLoad(path)

    % Convert the image to grayscale
    imageVector = rgb2gray(path);
    imshow(imageVector)
end


% Func to calculate Classical MDS and also Stress
function [embedding, totalStress] = classicalMDS(X, numDimensions)

    % Calculate similarity or dissimilarity matrix
    D = pdist(X);   % Euclidean distance between data points
    Dissimilarity = squareform(D);   % Convert to dissimilarity matrix
    
    % Center the dissimilarity matrix
    N = size(Dissimilarity, 1);
    J = eye(N) - ones(N) / N;
    B = -1/2 * J * Dissimilarity * J;
    
    % Perform eigenvalue decomposition
    [EigenVector, EigenValue] = eig(B);
    eigenvalues = diag(EigenValue);
    
    % Sort eigenvalues and eigenvectors in descending order
    [eigenvalues, indices] = sort(eigenvalues, 'descend');
    EigenVector = EigenVector(:, indices);
    
    % Select the desired number of dimensions
    numDimensions = 2;
    embedding = EigenVector(:, 1:numDimensions) * sqrt(EigenValue(1:numDimensions, 1:numDimensions));
    
    % Compute pairwise Euclidean distances in the embedding
    EmbeddingDistances = pdist(embedding);
    % Calculate stress for each individual point
    individualStress = (D - EmbeddingDistances).^2;
    % Compute total stress as the sum of individual stresses
    totalStress = sum(individualStress());
    % temp = individualStress/sum(Dissimilarity.^2)
    % Print the stress value for each point and the total stress
    disp('Total Stress:');
    disp(totalStress);
    % disp(temp)

    % Plot the MDS embedding
    figure;
    scatter(embedding(:, 1), embedding(:, 2), 'filled');
    colormap('jet');
    colorbar;
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    title('MDS Embedding');
end

