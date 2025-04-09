function practica6_hands_on_ANN_Regression
% Este script contiene la resolución del tutorial práctico del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

load('..\..\0_MATERIAL\Bases_de_datos\data\MAT\Boston');
% Nombre de las variables
var_names=Boston.Properties.VariableNames

% Dimensiones de la base de datos original
size(Boston)

% Dimensiones de la base de datos sin valores NaN
dim = size(Boston)

disp('%%%%%%%%%%%%%%% REDES NEURONALES ARTIFICIALES %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

Y=Boston.medv;
X=Boston{:,1:end-1};

rng(13);
net = fitnet(20); % 20 neuronas en capa oculta

rng(1);[trainInd,valInd,testInd] = dividerand(length(Y),0.7,0.15,0.15);
net.layers{1}.transferFcn = 'logsig'; % Sigmoid as transfer function between layers
net.divideFcn='divideind';
net.divideParam.trainInd=trainInd;
net.divideParam.valInd=valInd;
net.divideParam.testInd=testInd;
% net.inputs{1}.processFcns{2}='mapstd';
% net.outputs{1}.processFcns{2}='mapstd';
    
    
% Train ANN
train_net = train(net,X',Y');

% Test ANN 
ypred=train_net(X(testInd,:)');
MSE = mean((ypred-Y(testInd)').^2);
fprintf('RMSE de la RNA (nodos en capa oculta=%d) = %4.2f \n\n',20,sqrt(MSE));


% CALCULAR NÚMERO ÓPTIMO DE NEURONAS EN LA CAPA OCULTA
% Usar 5-fold CV en los datos de entrenamiento para elegir ALPHA
rng(2)
k = 5;
c = cvpartition(length(trainInd)+length(valInd),'KFold',k);

posCV = [trainInd valInd];posCV = sort(posCV);
X1 = X(posCV,:);
Y1 = Y(posCV);

CV_MSE=[];N_grid = [5 10 15 20 30 40 50];
for aa = 1:k
    pos_train = find(c.training(aa));
    pos_test_CV = find(c.test(aa));
    pos_val_CV = pos_train(1:length(pos_test_CV));
    pos_train_CV = pos_train(length(pos_test_CV)+1:end);
    
       
    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(N_grid) %Si hay M niveles de poda, hay M+1 alphas -> la última no cogemos sería poda completa -> decir clase mayoritaria
        rng(13);
        net = fitnet(N_grid(bb));
        
        net.layers{1}.transferFcn = 'logsig'; % Sigmoid as transfer function between layers
        net.divideFcn='divideind';
        net.divideParam.trainInd=pos_train_CV;
        net.divideParam.valInd=pos_val_CV;
        net.divideParam.testInd=pos_test_CV;
%         net.inputs{1}.processFcns{2}='mapstd';
%         net.outputs{1}.processFcns{2}='mapstd';
        
        % Train ANN
        train_net = train(net,X1',Y1');
        
        % Test ANN
        ypred=train_net(X1(pos_test_CV,:)');
        CV_MSE(aa,bb) = mean((ypred-Y1(pos_test_CV)').^2);
    end
    
end
[val,pos] = min(mean(CV_MSE));

errorbar(N_grid,mean(CV_MSE),std(CV_MSE));hold on;plot(N_grid,mean(CV_MSE),'ro');hold off;xlabel('#neuronas');ylabel('RMSE CV');
pause;

rng(13);
net = fitnet(N_grid(pos));

net.layers{1}.transferFcn = 'logsig'; % Sigmoid as transfer function between layers
net.divideFcn='divideind';
net.divideParam.trainInd=trainInd;
net.divideParam.valInd=valInd;
net.divideParam.testInd=testInd;
% net.inputs{1}.processFcns{2}='mapstd';
% net.outputs{1}.processFcns{2}='mapstd';
    
% Train ANN
train_net = train(net,X',Y');

% Test ANN 
ypred=train_net(X(testInd,:)');
MSE = mean((ypred-Y(testInd)').^2);
fprintf('RMSE de la RNA (nodos en capa oculta=%d) = %4.2f \n\n',N_grid(pos),sqrt(MSE));

