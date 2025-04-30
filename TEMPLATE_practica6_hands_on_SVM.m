function TEMPLATE_practica6_hands_on_SVM
% Este script contiene la resolución del tutorial práctico del Tema 6 (SVM)
% de la asignatura 'Técnicas de Inteligencia Artificial'

clc;
clear all;
close all;

disp('%%%%%%%%%%%%%%%%%%% SUPPORT VECTOR MACHINE %%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

%Generamos 2 predictores aleatorios normalmente distribuidos
rng(6)

len = 20;
x = randn(len, 2);
y = [-1*ones(len/2,1); ones(len/2,1)];
x(y == 1,:)=x(y == 1,:) + 1;

% Scatter
plot(x(y==-1,1),x(y==-1,2),'o','MarkerSize',6,'MarkerEdgeColor','b','MarkerFaceColor','b');
hold on;plot(x(y==1,1),x(y==1,2),'o','MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor','r');
xlabel('x(:,1)');ylabel('x(:,2)');title('SVC   C=10');v=axis;pause;

% Ajustamos SVM con kernel lineal (SVC) a los datos
% (1/BoxConstraint =~ C) pero a BoxConstraint lo llamamos "C" y le damos valor C=>10
SVMModel = fitcsvm(x, y, "BoxConstraint",10, "KernelFunction","linear")

% Dibujamos umbral de decisión
    % Beta y bias
    beta = SVMModel.Beta;
    bias = SVMModel.Bias;
    % Despejar x2
    x2_pred = -(bias + beta(1)*x(:,1))/beta(2);

    plot(x(:,1),x2_pred,'Linewidth',1.5);axis(v);
    pause;

% Dibujamos support vectors
% Remember we can double click the variables in debug mode to see what stuff they have! SVMModel has SupportVectors and IsSupportVector!
% We can use SupportVectors directly
% We can also do x(SVMModel.IsSupportVector)
sv = SVMModel.SupportVectors;
plot(sv(:,1),sv(:,2),'o','MarkerSize',10,'MarkerEdgeColor','k')
hold off;pause;close;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Repetimos proceso anterior con C = 0.1 -> margen más ancho -> más
% infracciones del margen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Usamos 10-FOLD CV para buscar el C óptimo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(2)
k = 10;
c = cvpartition(len,'KFold',k);

CV_error=[];
C_grid = [0.001,0.01,0.1,1.5,10,100];

for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = x(pos_train_CV,:);
    Xtest = x(pos_test_CV,:);
    Ytrain = y(pos_train_CV);
    Ytest = y(pos_test_CV);
      
    % Para cada C, ajustamos y evaluamos los modelos
    for bb=1:length(C_grid)
        SVMModel = fitcsvm(x, y, "BoxConstraint",C_grid(bb), "KernelFunction","linear");
        label = predict(SVMModel, Xtest);
        CV_error(aa, bb) = 100*(1-(sum(label == Ytest)/length(Ytest)));
    end
    
end
% Calculamos min de CV_error
[val, pos] = min(mean(CV_error))


% Entrenamos modelo con C seleccionada a través de CV
SVMModel = fitcsvm(x,y, "BoxConstraint",C_grid(pos), "KernelFunction","linear");

% Creamos una base de datos de test
rng(33);
xtest = randn(len,2);
ytest = [-1*ones(len/2,1); ones(len/2,1)];
xtest(ytest==1,:) = xtest(ytest==1,:) + 1;

% Evaluamos el modelo
label = predict(SVMModel, xtest);
acierto = 100*(sum(label==ytest)/length(ytest));

fprintf('Precisión del SVC (C=%.3f) = %.2f \n',C_grid(pos),acierto);

% Matriz de confusión
C = confusionmat(ytest,label);
confusionchart(C,{'Clase (-1)','Clase (1)'})
pause;
close;

% Comprobamos ahora que al cambiar el C óptimo los resultados en test
% empeoran un poco

% Entrenamos modelo

% Evaluamos el modelo


fprintf('Precisión del SVC (C=%.3f) = %.2f \n',C_grid(pos),acierto);

% Matriz de confusión
C = confusionmat(ytest,label);
confusionchart(C,{'Clase (-1)','Clase (1)'})
pause;close;


%% TODO ANTES DE ESTE PUNTO ES PROBLEMÁTICO - LOS ERRORES DAN MEJOR DE LO QUE DEBERÍAN!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM -> Umbrales de decisión no lineales
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(1);

len = 200;
x = randn(len,2);

% Los primeros 150 son de clase 1, los ultimos 50 de clase 2
y = [ones(150, 1);2*ones(50,1)];
x(1:100,:) = x(1:100,:) + 2;
x(101:150,:) = x(101:150,:) - 2;

% Scatter
plot(x(y==1,1),x(y==1,2),'o','MarkerSize',6,'MarkerEdgeColor','b','MarkerFaceColor','b');
hold on;
plot(x(y==2,1),x(y==2,2),'o','MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor','r');

xlabel('x(:,1)');
ylabel('x(:,2)');

v=axis;
pause;

% Ajustamos SVM con kernel Gaussiano
% De las diapositivas: Gamma = 1/(KernelScale²)
SVMModel = fitcsvm(x, y, "BoxConstraint",1, "KernelFunction","gaussian","KernelScale",1);

% Ploteamos umbral de decisión
% 1) Construimos rejilla
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(x(:,1)):d:max(x(:,1)), min(x(:,2)):d:max(x(:,2)));

% 2) Usamos modelo ajustado para predecir
xGrid = [x1Grid(:), x2Grid(:)];
[~,scores]=predict(SVMModel, xGrid)

% 3) Visualizamos support vector y umbral de decisión
plot(x(SVMModel.IsSupportVector,1),x(SVMModel.IsSupportVector,2),'o','MarkerSize',10,'MarkerEdgeColor','k');
% contour

contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)),[0 0],'k')

title('SVM   C=1   KS=1');
hold off;pause;close;


% Dividimos, ahora, la base de datos en train y test
rng(1); % Fijamos semilla para el generado de números aleatorios
hpartition = cvpartition(len,'Holdout',0.50); % Partición no estratificada
% 50% train y 50% test
pos_train = hpartition.training;
pos_test = hpartition.test;

% Ajustamos SVM con kernel Gaussiano C=1, gamma=1 en TRAIN
SVMModel = fitcsvm(x(pos_train,:), y(pos_train), "BoxConstraint",1, "KernelFunction","gaussian","KernelScale",1);

% Evaluamos el modelo en TEST
label = predict(SVMModel,x(pos_test,:));
acierto = 100*sum(label==y(pos_test))/length(y(pos_test));
fprintf('Precisión de la SVM (C=%.3f  KS=%.3f) = %.2f \n',1,1, acierto);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Usamos 10-FOLD CV para buscar el par (C,KS) óptimos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(2)
k = 10;
c = cvpartition(sum(pos_train),'KFold',k);

x1 = x(pos_train,:);
y1 = y(pos_train);
x2 = x(pos_test,:);
y2 = y(pos_test);

CV_error=[];

C_grid = [0.1,1,10,100,1000];
KS_grid = [0.5 1 2 3 4];

for aa = 1:k
    pos_train_CV = c.training(aka);
    pos_test_CV = c.test(aa);

    Xtrain = x1(pos_train_CV,:);
    Xtest = x1(pos_test_CV,:);
    Ytrain = y1(pos_train_CV);
    Ytest = y1(pos_test_CV);
      
    % Para cada combinación C - KS ajustamos y evaluamos los modelos
    for bb=1:length(C_grid)
        for cc=1:length(KS_grid)
            SVMModel_CV = fitcsvm(Xtrain, Ytrain, "BoxConstraint",C_grid(bb), "KernelFunction","gaussian","KernelScale",KS_grid(cc));

            % Evaluamos el modelo en TEST
            label = predict(SVMModel_CV,Xtest);
            CV_error(bb,cc,aa) = 100*(1- sum(label==Ytest)/length(Ytest));
        end
    end
    
end
% buscamos CV_error mínimo
CV_medios = mean(CV_error,3);
[val, pos] = min(CV_medios(:));
[row, col] = ind2sub(size(CV_medios),pos);

% Entrenamos modelo con C seleccionada a través de CV
SVMModel = fitcsvm(x1,y1,'BoxConstraint',C_grid(row),'KernelFunction','gaussian','KernelScale',KS_grid(col));

% Evaluamos el modelo
[label,scores] = predict(SVMModel,x2);
acierto = 100*sum(label==y2)/length(y2);
fprintf('Precisión de la SVM (C=%.3f  KS=%.3f) = %.2f \n',C_grid(row),KS_grid(col),acierto);

C = confusionmat(y2,label);
confusionchart(C,{'Clase (1)','Clase (2)'})
pause;close;

% PLoteamos curva ROC

% X = False positive rate
% Y = True positive rate (Especificidad)
[X, Y, T, AUC, OPTROCPT] = perfcurve(y2, scores(:,2),2);

plot(X,Y)
xlabel('1 - Especificidad') 
ylabel('Sensibilidad')
title(sprintf('AUC = %.2f',AUC));
pause;
close;