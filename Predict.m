function [ predict_Label, predict_F ] = Predict( X, W )
% predict the linear model

    num_instance = size(X, 1);
    num_class = size(W, 2);
    
    predict_F = X * W;
    threshold = 0;
    predict_Label = double(predict_F > threshold);
    predict_Label(predict_Label < 1) = -1;
    
    for j = 1: num_instance
        if sum(predict_Label(j, :)) == -num_class
            max_column_index = find(predict_F(j, :) == max(predict_F(j, :)), 1);
            predict_Label(j, max_column_index) = 1;
        end
    end
end