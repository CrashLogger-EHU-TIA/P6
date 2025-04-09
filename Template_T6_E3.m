function Template_T6_E3
% Este script contiene la resolución del ejercicio aplicado 3 del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% En este problema usaremos SVC y SVM para predecir si un determinado coche 
% tiene alta o baja autonom´ıa en la base de datos Auto. 

% Cargamos base de datos


disp('%%%%%%%%%%%%%%%%% EJERCICIO 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Remover valores NaN


disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 -  Crear una variable binaria que tome el valor 1 para coches 
% con una autonomía (mpg) superior a la mediana, y que tome el valor 0 
% para coches con una autonomía inferior a la mediana.




fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar una red neuronal artificial a los datos con varios 
% valores de H (número de neuronas en la capa oculta de la red) con el objetivo 
% de predecir si un coche tiene alta o baja autonom´ıa. Reportar el error 
% CV asociado a los diferentes valores del parámetro H. Comentar los resultados.
rng(13);
 % Crear red neuronal con 20 neuronas en capa oculta

rng(7);
% Dividir datos en train (0.7), validation (0.15) y test (0.15) 

    




% CALCULAR NÚMERO ÓPTIMO DE NEURONAS EN LA CAPA OCULTA
% Usar 5-fold CV en los datos de entrenamiento para elegir H
rng(2)
k = 5;
