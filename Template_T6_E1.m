function Template_T6_E1
% Este script contiene la resolución del ejercicio aplicado 1 del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% En la práctica aplicamos un árbol de clasificación a la base de datos 
% Carseats después de convertir la variable respuesta Sales en una variable
% cualitativa. Ahora trataremos de predecir Sales usando árboles de regresión
% y aproximaciones relacionadas, tratando la respuesta como una variable 
% cuantitativa. 

clear all;
close all;
clc;

% Cargamos base de datos
load Carseats.mat;

disp('%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Divide los datos de manera aleatoria en conjuntos de 
% entrenamiento (50%) y de test (50%). Fijar la semilla para la 
% generación de números pseudo-aleatorios a rng(4).
rng(4); % Fijamos semilla para el generado de números aleatorios

hpartition = cvpartition(size(Carseats,1),"HoldOut",0.5);
pos_train = hpartition.training;
pos_test = hpartition.test;
Y = Carseats.Sales;
X = Carseats(:,2:end);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un árbol de regresión usando el conjunto de 
% entrenamiento. Visualiza el árbol e interpreta los resultados. 
% ¿Cuál es el MSE de test obtenido?
% Entrenamos árbol de regresión

tree = fitrtree(X(pos_train,:),Y(pos_train));
alpha_grid = tree.PruneAlpha;

% El "Plot"
view(tree,'Mode','graph')

ypred = predict(tree,X(pos_test,:));
MSE_test = mean((Y(pos_test)-ypred).^2);

fprintf("MSE de test: %4.4f\n", MSE_test)

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Utiliza CV para determinar el nivel óptimo de complejidad 
% del árbol. ¿Mejora el MSE de test con la poda del árbol?
rng(2);

% Como en el tema 5, hacemos kfold
k = 10;
c = cvpartition(sum(pos_train),'KFold',k);
X1 = X(pos_train,:);
Y1 = Y(pos_train);

clear("CV_MSE");
CV_MSE=[];
for aa = 1:k
    pos_train_iter = c.training(aa);
    pos_test_iter = c.test(aa);
    
    Xtrain = X1(pos_train_iter,:);
    Xtest = X1(pos_test_iter,:);
    Ytrain = Y1(pos_train_iter);
    Ytest = Y1(pos_test_iter);

    tree_iter = fitrtree(Xtrain,Ytrain);

    % Buscamos el nivel de poda optimo
    for bb=1:length(alpha_grid)-1 
        tree2 =prune(tree_iter,"Alpha",alpha_grid(bb));
        ypred = predict(tree2,Xtest);
        CV_MSE(aa,bb) = mean((ypred-Ytest).^2);
    end

end

% El nivel de poda optimo entonces:
[val,pos] = min(mean(CV_MSE));
tree_pruned = prune(tree, "Alpha", alpha_grid(pos));

view(tree_pruned,'Mode','graph')

% El MSE del arbol podado no es el VAL de CV_MSE! En ese el Xtest era una subdivisión de la mitad de training de la DB original!
ypred = predict(tree_pruned,X(pos_test,:)); 
MSE_test_pruned = mean((ypred-Y(pos_test)).^2);

fprintf("MSE de test: %4.4f\n", MSE_test_pruned);
fprintf("El mse de test mejora con el arbol podado: evitamos sobreajustar el modelo a los datos.")

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('Dale a ENTER para continuar (Los graficos de los arboles no se cierran con close all)')
pause
% Apartado 4 - Utiliza bagging para analizar los datos. ¿Cuál es el MSE de 
% test obtenido?  Determina qué variables son las más importantes.
rng(4);

mdl_bagged = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample","all", "Method","regression", 'OOBPredictorImportance', 'on');
view(mdl_bagged.Trees{1},'Mode','graph')

% Importancia normalizada
imp = mdl_bagged.OOBPermutedPredictorDeltaError;
imp = imp/max(imp);

figure();
bar(imp);
ylabel('Importancia');
xlabel('Variables');

h = gca;
h.XTickLabel = mdl_bagged.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

figure()
plot(oobError(mdl_bagged));
xlabel('Arboles');
ylabel('Out Of Bag MSE');


ypred = predict(mdl_bagged, X(pos_test,:));
MSE = mean((ypred-Y(pos_test)).^2);
fprintf('MSE de test del árbol bagged = %4.2f \n\n',MSE);
fprintf('Las variables mas importantes son las primeras del arbol (las de mas arriba). En este caso, queda claro que ShelveLoc (Posición en las estanterías de la tienda) y Price (El precio de las sillitas de bebé) son las variables mas importantes al determinal las ventas de un modelo específico\n')

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Utiliza random forests para analizar los datos. ¿Cuál es el 
% MSE de test obtenido?  Determina qué variables son las más importantes. 
% Describe el efecto de m, el número de predictores considerado en cada 
% división, en la tasa de error obtenida.
rng(4);

m = 3;
mdl_RF = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",m, "Method","regression");

ypred = predict(mdl_RF,X(pos_test,:));
MSE_RF = mean((ypred-Y(pos_test)).^2);
fprintf('MSE TEST del RF con m=%d = %4.2f \n\n',m,MSE_RF);
fprintf('')

% Analizamos el efecto de m usando CV
rng(4);

k = 10;
m = linspace(1,size(X1,2),size(X1,2));

clear("CV_MSE");
CV_MSE=[];
for aa = 1:k
    pos_train_iter = c.training(aa);
    pos_test_iter = c.test(aa);
    
    Xtrain = X1(pos_train_iter,:);
    Xtest = X1(pos_test_iter,:);
    Ytrain = Y1(pos_train_iter);
    Ytest = Y1(pos_test_iter);

    mdl_RF = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",aa, "Method","regression");

    % Buscamos el nivel de poda optimo
    for bb=1:length(m) 
        tree3 = TreeBagger(100,Xtrain, Ytrain, "NumPredictorsToSample",m(bb),"Method","regression");
        ypred = predict(tree3,Xtest);
        CV_MSE(aa,bb) = mean((ypred-Ytest).^2);
    end

end

[val,pos] = min(mean(CV_MSE));
tree_random_forest = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample", pos, "Method","regression");

ypred = predict(mdl_RF,X(pos_test,:));
MSE_RF = mean((ypred-Y(pos_test)).^2);
fprintf('MSE TEST del RF con m=%d = %4.2f \n\n', pos, MSE_RF);

figure();
errorbar(m(1:end),mean(CV_MSE),std((CV_MSE)));
hold on;
plot(m(1:end),mean(CV_MSE),'ro');
hold off;
title('MSE par diferentes m');
xlabel('m');
ylabel('MSE de Cross Validation');