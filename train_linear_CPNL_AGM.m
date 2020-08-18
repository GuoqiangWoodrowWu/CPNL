function [ W_alpha_1, obj ] = train_linear_CPNL_AGM( X, Y, lambda_1, lambda_2, lambda_3, NIter )
% Summary of this function: To train the linear CPNL by AGM
% Input: size(X) = [n_instances, n_features] 
%        size(Y) = size(n_instances, n_labels)
%        Y \in {-1, +1}
%        lambda_1: for the F-norm regularizer
%        lambda_2: for the positive label correlation regularizer
%        lambda_3: for the negative label correlation regularizer     
%        NIter: the max iteration number of AGM
% Output: size(W) = [n_features, n_labels]
% Written by Guoqiang Wu

    % Calculate the cost matrix and laplacian matrix
    C = Preprocess_Cost_Sensitive(Y);
    L_pos = Calculate_Laplacian_Positive(Y);
    L_neg = Calculate_Laplacian_Negative(Y);
    
    [num_instance, num_feature] = size(X);
    num_class = size(Y, 2);
    
    W_beta = zeros(num_feature, num_class);
    W_alpha_1 = zeros(num_feature, num_class);
    W_alpha_0 = zeros(num_feature, num_class);
    V = zeros(num_feature, num_class);
    
    j = 1;
    epsilon = 10^-6;
    lipschitz_1 = sqrt(4 * (norm(X, 'fro')^2)^2 * max(max(C))^2 + 4 * lambda_1^2 + ...
        4 * lambda_2^2 * (norm((X'*X), 'fro')^2 * norm(L_pos, 'fro')^2) + ...
        4 * lambda_3^2 * (norm((X'*X), 'fro')^2 * norm(L_neg, 'fro')^2));
    t_1 = 1;
    while ((j <= 2 || abs(obj(j - 1) - obj(j - 2)) / obj(j - 2) > epsilon) && j < NIter)
        % Calculate least square hinge loss subgradient
        [ V ] = Calculate_Subgradient( X, Y, W_beta, lambda_1, lambda_2, lambda_3, C, L_pos, L_neg );
        
        W_alpha_0 = W_alpha_1;
        W_alpha_1 = W_beta - 1 / (lipschitz_1) * V; 
        
        t_0 = t_1;
        t_1 = (1 + sqrt(1 + 4 * t_1^2)) / 2;
        W_beta = W_alpha_1 + (t_0 - 1) / t_1 * (W_alpha_1 - W_alpha_0);
                
        % Calculate the objective function value
        obj(j) = fValue( X, Y, W_alpha_1, lambda_1, lambda_2, lambda_3, C, L_pos, L_neg );
        j = j + 1;
    end
    % plot(obj);
end

function [ f_value ] = fValue( X, Y, W, lambda_1, lambda_2, lambda_3, C, L, L_neg )
    [num_instance, num_class] = size(Y);
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    temp = max(Z, I - Y .* (X * W));
    f_value = 0.5 * sum(sum(temp .* temp .* C, 2)) + 0.5 * lambda_1 * norm(W, 'fro')^2;
    f_value = 0.5 * f_value + 0.5 * lambda_2 * trace(X * W * L * W' * X');
    f_value = f_value + 0.5 * lambda_3 * trace(X * W * L_neg * W' * X');
end

function [ V ] = Calculate_Subgradient( X, Y, W, lambda_1, lambda_2, lambda_3, C, L, L_neg )
    % Calculate least square hinge loss subgradient
    [num_instance, num_class] = size(Y);
    
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    grad = X' * (-Y .* max(Z, I - Y .* (X * W)) .* C);
    V = grad + lambda_1 * W;

    V = V + lambda_2 * X' * X * W * L;
    V = V + lambda_3 * X' * X * W * L_neg;
end


