function TEMPLATE_practica6_hands_on_Trees_Regression
% Este script contiene la resolución del tutorial práctico relacionado con 
% ÁRBOLES DE REGRESIÓN del Tema 6 de la asignatura 'Técnicas de Inteligencia Artificial'
close all;clc;
% Cargamos base de datos


% Dimensiones de la base de datos original


disp('%%%%%%%%%%%%%%%%%%%% ÁRBOLES DE REGRESIÓN %%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

rng(4); % Fijamos semilla para la generación de números aleatorios
% Dividimos la base de datos en 50% train y 50% test

% Posiciones de observaciones de entrenamiento

% Posiciones de observaciones de test


% Respuesta

% Predictores


% Entrenamos árbol de regresión
tree = 0;
% Almacenamos grid de parámetro de regularización alpha
alpha_grid = 0;
% Visualizamos árbol de clasificación
 
% Evaluamos rendimiento del árbol de clasificación en test
% Respùestas estimadas

% MSE

% Visualizamos en línea de comandos RMSE
fprintf('RMSE del árbol de regresión (nodos terminales=%d) = %4.2f \n\n',sum(~tree.IsBranchNode),sqrt(MSE));


% PODA DEL ÁRBOL
rng(2);
% Usar K-fold CV en los datos de entrenamiento para elegir ALPHA óptima


% Generamos variables X1 e Y1 que almacenen predictores y respuesta de
% TRAIN, respectivamente
X1 = 0;
Y1 = 0;

% Bucle FOR para realizar 10-FOLD CV y optimizar ALPHA
CV_MSE=[];
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = X1(pos_train_CV,:);
    Xtest = X1(pos_test_CV,:);
    Ytrain = Y1(pos_train_CV);
    Ytest = Y1(pos_test_CV);
    
    % Entrenamos árbol
    
    
    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(alpha_grid)-1 %Si hay M niveles de poda, hay M+1 alphas -> la última no cogemos sería poda completa -> decir clase mayoritaria

    end
    
end
% Calculamos MIN de las medias de CV_MSE

% Podamos T0 con ALPHA óptimo
tree_pruned = 0;
% Visualizamos árbol de clasificación
view(tree_pruned,'Mode','graph');

% Calculamos CV_RMSE
CV_RMSE = [];
% Evaluamos rendimiento del árbol de clasificación podado en test
% Predicciones

% MSE

% Visualizamos en línea de comandos RMSE
fprintf('RMSE (TEST) del árbol podado (alpha=%.3f  nodos terminales=%d) = %4.2f \n\n',alpha_grid(pos),sum(~tree_pruned.IsBranchNode),sqrt(MSE));

% Errorbar -> alpha_grid vs mean(RMSE)

hold on;plot(alpha_grid(1:end-1),mean(CV_RMSE),'ro');hold off;xlabel('alpha');ylabel('RMSE CV');
pause;close;

rng(4);
% PODA DEL ÁRBOL
% Usar K-fold CV en los datos de entrenamiento pero usando función cvloss

% Podamos T0 con ALPHA óptimo
tree_pruned = 0;
% Visualizamos árbol de clasificación
view(tree_pruned,'Mode','graph')

% Evaluamos rendimiento del árbol de clasificación podado en test
% Predicciones

% MSE

% Visualizamos en línea de comandos RMSE
fprintf('RMSE (TEST) del árbol podado (cvloss, nodos terminales=%d) = %4.2f\n\n',sum(~tree_pruned.IsBranchNode),sqrt(MSE));

% Errorbar -> Nleaf vs mean(RMSE)

hold on;plot(Nleaf,E,'ro');hold off;xlabel('#nodos terminales');ylabel('MSE CV');





disp('%%%%%%%%%%%%%%%%%%% BAGGING/RANDOM FORESTS %%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

rng(4);
% Ajustamos BAGGING
mdl_bagged = 0;
% Visualizamos el primero de los árboles de regresión
view(mdl_bagged.Trees{1},'Mode','graph')

% Evaluamos rendimiento en test del árbol bagged 
% Predicciones

% MSE

% Visualizamos en línea de comandos RMSE
fprintf('RMSE (TEST) del árbol bagged = %4.2f \n\n',sqrt(MSE));


rng(4);
% Ajustamos RF
mdl_RF = 0;
% Visualizamos el primero de los árboles de regresión
view(mdl_RF.Trees{1},'Mode','graph')

% Evaluamos rendimiento en test del RF 
% Predicciones

% MSE

% Visualizamos en línea de comandos RMSE
fprintf('RMSE (TEST) del RF = %4.2f \n\n',sqrt(MSE));


% RF -> importancia de los predictores y OOB error
rng(4);
% Ajustamos RF con opción 'OOBPredictorImportance' a 'on'
mdl_RF_OOB = 0;
% Almacenamos la importancia de las variables 'OOBPermutedPredictorDeltaError'

% Creamos figura
figure;
% Dibujamos la IMPORTANCIA en diagrama de barras

ylabel('Importancia');
xlabel('Predictores');
h = gca;
h.XTickLabel = mdl_RF_OOB.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% Visualizamos #árboles vs OOB error

xlabel('Número de árboles');
ylabel('OOB RMSE');

% Visualizamos el error del RF con todos los árboles (función ooberror)

fprintf('RMSE OOB del RF = %4.2f \n\n',sqrt(err));




disp('%%%%%%%%%%%%%%%%%%%%%%%%%% BOOSTING %%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
rng(4);
% Ajustamos ensemble de 100 árboles usando BOOSTING
mdl_boosted = [];

% Evaluamos rendimiento en test del árbol bagged 
% Predicciones

% MSE

% Visualizamos en línea de comandos RMSE
fprintf('RMSE (TEST) del boosted tree (B=100) = %4.2f \n\n',sqrt(MSE));
