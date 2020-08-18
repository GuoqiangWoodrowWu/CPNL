function [ L ] = Calculate_Laplacian_Positive( Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    Y(Y < 0) = 0;
    [num_instance, num_label] = size(Y);
    % Calculate the similarity matrix
    S = zeros(num_label, num_label);
    
    % hamming distance similarity
    for i = 1: num_label
        for j = 1: num_label
            if i == j
                S(i, j) = 0;
            else
                S(i, j) = sum(Y(:, i) == Y(:, j)) / num_instance;
            end
        end
    end

    % k nearest neighbor
    k = 3;
    B = sort(S, 2, 'descend');
    b = B(:, k);
    for i = 1: num_label
        for j = 1: num_label
            if i == j || S(i, j) <= b(i)
                S(i, j) = 0;
            end
        end
    end
    S = (S + S')/2;
    
    % Calculate the laplacian matrix
    D = diag(sum(S));
    L = D - S;
end

