function TEMPLATE_practica6_hands_on_SVM
% Este script contiene la resolución del tutorial práctico del Tema 6 (SVM)
% de la asignatura 'Técnicas de Inteligencia Artificial'

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
