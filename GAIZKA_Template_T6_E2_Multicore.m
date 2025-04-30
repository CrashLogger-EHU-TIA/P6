function Template_T6_E2_Multicore
% Multicore-optimized version of the SVM exercise

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool; % Uses default number of workers (usually equal to physical cores)
end

% Load data
load Auto.mat;

% Preprocessing (same as before)
Auto = rmmissing(Auto);

% Create binary variable
autonomia(Auto.mpg>median(Auto.mpg)) = 1;
autonomia(Auto.mpg<median(Auto.mpg)) = 0;
autonomia = autonomia';

x = table2array(Auto(:,2:8)); % Convert table to array
y = autonomia;

% Set up cross-validation
rng(1);
hpartition = cvpartition(length(y),'Holdout',0.50);
pos_train = hpartition.training;
pos_test = hpartition.test;

x1 = x(pos_train,:);
x2 = x(pos_test,:);
y1 = y(pos_train);
y2 = y(pos_test);

% 10-FOLD CV for linear SVC
rng(2)
k = 10;
c = cvpartition(length(y1),'KFold',k);

C_grid = linspace(0.1,2,10);
CV_error = zeros(k, length(C_grid));

% Parallelize the outer loop
parfor aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = x1(pos_train_CV,:);
    Xtest = x1(pos_test_CV,:);
    Ytrain = y1(pos_train_CV);
    Ytest = y1(pos_test_CV);
      
    temp_error = zeros(1, length(C_grid));
    for bb = 1:length(C_grid)
        SVMModel = fitcsvm(Xtrain, Ytrain, "BoxConstraint", C_grid(bb), ...
                          "KernelFunction", "linear");
        label = predict(SVMModel, Xtest);
        temp_error(bb) = 100*(1-sum(label==Ytest)/length(Ytest));
    end
    CV_error(aa,:) = temp_error;
end

% Find best C
[val, pos] = min(mean(CV_error));

% Train final model with best C
SVMModel = fitcsvm(x1, y1, "BoxConstraint", C_grid(pos), ...
                  "KernelFunction", "linear");

% Evaluate
label = predict(SVMModel, x2);
acierto = 100*(sum(label==y2)/length(y2));
fprintf('Precisión del SVC (C=%.3f) = %.2f \n', C_grid(pos), acierto);

% Confusion matrix
C = confusionmat(y2, label);
confusionchart(C, {'Clase (0)', 'Clase (1)'})
pause; close;

% Radial basis function SVM with parallel optimization
C_grid = linspace(0.01, 5, 200); % Reduced for demonstration
KS_grid = linspace(10, 100, 200); % Reduced for demonstration
CV_error = zeros(length(C_grid), length(KS_grid), k);

% Parallelize the outer loop
parfor aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = x1(pos_train_CV,:);
    Xtest = x1(pos_test_CV,:);
    Ytrain = y1(pos_train_CV);
    Ytest = y1(pos_test_CV);
      
    temp_error = zeros(length(C_grid), length(KS_grid));
    for bb = 1:length(C_grid)
        for cc = 1:length(KS_grid)
            SVMModel = fitcsvm(Xtrain, Ytrain, "BoxConstraint", C_grid(bb), ...
                              "KernelFunction", "gaussian", ...
                              "KernelScale", KS_grid(cc));
            label = predict(SVMModel, Xtest);
            temp_error(bb, cc) = 100*(1-sum(label==Ytest)/length(Ytest));
        end
    end
    CV_error(:,:,aa) = temp_error;
end

% Find best parameters
CV_medios = mean(CV_error, 3);
[val, pos] = min(CV_medios(:));
[row, col] = ind2sub(size(CV_medios), pos);

% Train final model
SVMModel = fitcsvm(x1, y1, 'BoxConstraint', C_grid(row), ...
                  'KernelFunction', 'gaussian', ...
                  'KernelScale', KS_grid(col));

% Evaluate
[label, scores] = predict(SVMModel, x2);
acierto = 100*sum(label==y2)/length(y2);
fprintf('Precisión de la SVM (C=%.3f  KS=%.3f) = %.2f \n', ...
        C_grid(row), KS_grid(col), acierto);

% Confusion matrix
C = confusionmat(y2, label);
confusionchart(C, {'Clase (1)', 'Clase (2)'})
pause; close;