function [ A_alpha_1, obj ] = train_kernel_CPNL_AGM( X, Y, lambda_1, lambda_2, lambda_3, sigma, NIter )
% Summary of this function: to train the kernel CPNL by AGM
% Input: size(X) = [n_instances, n_features] 
%        size(Y) = size(n_instances, n_labels)
%        Y \in {-1, +1}
%        lambda_1: for the F-norm regularizer
%        lambda_2: for the positive label correlation regularizer
%        lambda_3: for the negative label correlation regularizer
%        sigma: kernel function hyper-parameter     
%        NIter: the max iteration number of AGM
% Output: size(A) = [n_instances, n_labels]
% Written by Guoqiang Wu


    % Calculate the cost matrix and laplacian matrix
    C = Preprocess_Cost_Sensitive(Y);
    %C = ones(size(Y));
    L_pos = Calculate_Laplacian_Positive(Y);
    
    L_neg = Calculate_Laplacian_Negative(Y);
    
    [num_instance, num_feature] = size(X);
    num_class = size(Y, 2);
    
    %A = zeros(num_instance, num_class);
    A_alpha_0 = zeros(num_instance, num_class);
    A_alpha_1 = zeros(num_instance, num_class);
    A_beta = zeros(num_instance, num_class);
    V = zeros(num_instance, num_class);
    
    % Calculate Kernel matrix
    % rbf kernel
    K = zeros(num_instance, num_instance);
    for i = 1: num_instance
        for j = 1: num_instance
            K(i, j) = exp(-sigma*norm(X(i,:) - X(j,:), 2)^2);
        end
    end
    
    j = 1;
    epsilon = 10^-6;
    lipschitz_1 = sqrt(4 * (norm(K, 'fro')^2)^2 * max(max(C))^2 + 4 * lambda_1^2 * (norm(K, 'fro')^2)... 
        + 4 * lambda_2^2 * (norm((K'*K), 'fro')^2 * norm(L_pos, 'fro')^2) + ...
        4 * lambda_3^2 * (norm((K'*K), 'fro')^2 * norm(L_neg, 'fro')^2));
    t_1 = 1;
    while ((j <= 2 || abs(obj(j - 1) - obj(j - 2)) / obj(j - 2) > epsilon) && j < NIter)
        % Calculate least square hinge loss subgradient
        [ V ] = Calculate_Subgradient( K, Y, A_beta, lambda_1, lambda_2, lambda_3, C, L_pos, L_neg );
        A_alpha_0 = A_alpha_1;
        A_alpha_1 = A_beta - 1 / (lipschitz_1) * V; 
        
        t_0 = t_1;
        t_1 = (1 + sqrt(1 + 4 * t_1^2)) / 2;
        A_beta = A_alpha_1 + (t_0 - 1) / t_1 * (A_alpha_1 - A_alpha_0);
        % Calculate the objective function value
        obj(j) = fValue( Y, A_alpha_1, lambda_1, lambda_2, lambda_3, C, L_pos, K, L_neg );
        j = j + 1;
    end
    % plot(obj);
end

function [ f_value ] = fValue( Y, A, lambda_1, lambda_2, lambda_3, C, L, K, L_neg )
    [num_instance, num_class] = size(Y);
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    temp = max(Z, I - Y .* (K * A));
    f_value = 0.5 * sum(sum(temp .* temp .* C, 2)) + 0.5 * lambda_1 * trace(A' * K * A);
    f_value = f_value + 0.5 * lambda_2 * trace(K * A * L * A' * K');
    f_value = f_value + 0.5 * lambda_3 * trace(K * A * L_neg * A' * K');
end

function [ V ] = Calculate_Subgradient( K, Y, A, lambda_1, lambda_2, lambda_3, C, L, L_neg )
    [num_instance, num_class] = size(Y);
    
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    grad = K' * (-Y .* max(Z, I - Y .* (K * A)) .* C);
    V = grad + lambda_1 * (K * A);

    V = V + lambda_2 * K' * K * A * L;
    V = V + lambda_3 * K' * K * A * L_neg;
end


