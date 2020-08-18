function [ C ] = Preprocess_Cost_Sensitive( Y )
%UNTITLED Summary of this function goes here
%   Input : Y = [num_instance, num_label]
%   Output: C = [num_instance, num_label]
    
    [num_instance, num_label] = size(Y);
    C = zeros(num_instance, num_label);
    sum_tmp = sum(Y);
    pos_num = (sum_tmp + num_instance) / 2;
    neg_num = num_instance - pos_num;
    for i = 1: num_instance
        for j = 1: num_label
            if Y(i, j) == 1
                C(i, j) = neg_num(j) / pos_num(j);
            else
                C(i, j) = 1;
            end
        end
    end
    beta = 0.5;
    C = C.^(beta);
end