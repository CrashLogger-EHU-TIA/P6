function TEMPLATE_practica6_hands_on_SVM
% Este script contiene la resolución del tutorial práctico del Tema 6 (SVM)
% de la asignatura 'Técnicas de Inteligencia Artificial'

disp('%%%%%%%%%%%%%%%%%%% SUPPORT VECTOR MACHINE %%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

%Generamos 2 predictores aleatorios normalmente distribuidos
rng(6)

% Scatter
plot(x(y==-1,1),x(y==-1,2),'o','MarkerSize',6,'MarkerEdgeColor','b','MarkerFaceColor','b');
hold on;plot(x(y==1,1),x(y==1,2),'o','MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor','r');
xlabel('x(:,1)');ylabel('x(:,2)');title('SVC   C=10');v=axis;pause;

% Ajustamos SVM con kernel lineal (SVC) a los datos C=10

% Dibujamos umbral de decisión
    % Beta y bias

    % Despejar x2
    x2_pred [];

    plot(x(:,1),x2_pred,'Linewidth',1.5);axis(v);
    pause;

% Dibujamos support vectors
sv = [];
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

CV_error=[];C_grid = [0.001,0.01,0.1,1.5,10,100];
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = x(pos_train_CV,:);
    Xtest = x(pos_test_CV,:);
    Ytrain = y(pos_train_CV);
    Ytest = y(pos_test_CV);
      
    % Para cada C, ajustamos y evaluamos los modelos
    for bb=1:length(C_grid)
        
    end
    
end
% Calculamos min de CV_error


% Entrenamos modelo con C seleccionada a través de CV


% Creamos una base de datos de test
rng(33);


% Evaluamos el modelo


fprintf('Precisión del SVC (C=%.3f) = %.2f \n',C_grid(pos),acierto);

% Matriz de confusión
C = confusionmat(ytest,label);
confusionchart(C,{'Clase (-1)','Clase (1)'})
pause;close;

% Comprobamos ahora que al cambiar el C óptimo los resultados en test
% empeoran un poco

% Entrenamos modelo

% Evaluamos el modelo


fprintf('Precisión del SVC (C=%.3f) = %.2f \n',C_grid(pos),acierto);

% Matriz de confusión
C = confusionmat(ytest,label);
confusionchart(C,{'Clase (-1)','Clase (1)'})
pause;close;








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM -> umbrales de decisión no lineales
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1);
% creamos 200 observaciones de 2 predictores aleatorios normalmente distribuidos

% Los primeros 150 son de clase -1 y los últimos 150 de clase 1

% Scatter
plot(x(y==1,1),x(y==1,2),'o','MarkerSize',6,'MarkerEdgeColor','b','MarkerFaceColor','b');
hold on;plot(x(y==2,1),x(y==2,2),'o','MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor','r');
xlabel('x(:,1)');ylabel('x(:,2)');v=axis;pause;

% Ajustamos SVM con kernel Gaussiano


% Ploteamos umbral de decisión
% 1) Construimos rejilla

% 2) Usamos modelo ajustado para predecir

% 3) Visualizamos support vector y umbral de decisión
plot(x(SVMModel.IsSupportVector,1),x(SVMModel.IsSupportVector,2),'o','MarkerSize',10,'MarkerEdgeColor','k');
% contour

title('SVM   C=1   KS=1');
hold off;pause;close;


% Dividimos, ahora, la base de datos en train y test
rng(1); % Fijamos semilla para el generado de números aleatorios
hpartition = cvpartition(len,'Holdout',0.50); % Partición no estratificada
% 50% train y 50% test
pos_train = hpartition.training;
pos_test = hpartition.test;

% Ajustamos SVM con kernel Gaussiano C=1, gamma=1 en TRAIN

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

CV_error=[];C_grid = [0.1,1,10,100,1000];KS_grid = [0.5 1 2 3 4];
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = x1(pos_train_CV,:);
    Xtest = x1(pos_test_CV,:);
    Ytrain = y1(pos_train_CV);
    Ytest = y1(pos_test_CV);
      
    % Para cada combinación C - KS ajustamos y evaluamos los modelos
    for bb=1:length(C_grid)
        for cc=1:length(KS_grid)


        end
    end
    
end
% buscamos CV_error mínimo


% Entrenamos modelo con C seleccionada a través de CV
SVMModel = fitcsvm(x1,y1,'BoxConstraint',C_grid(row),'KernelFunction','gaussian',...
    'KernelScale',KS_grid(col));

% Evaluamos el modelo
[label,scores] = predict(SVMModel,x2);
acierto = 100*sum(label==y2)/length(y2);
fprintf('Precisión de la SVM (C=%.3f  KS=%.3f) = %.2f \n',C_grid(row),KS_grid(col),acierto);

C = confusionmat(y2,label);
confusionchart(C,{'Clase (1)','Clase (2)'})
pause;close;

% PLoteamos curva ROC

plot(X,Y)
xlabel('1 - Especificidad') 
ylabel('Sensibilidad')
title(sprintf('AUC = %.2f',AUC));
pause;close;