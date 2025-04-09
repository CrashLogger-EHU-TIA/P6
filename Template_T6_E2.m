function Template_T6_E2
% Este script contiene la resolución del ejercicio aplicado 2 del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% En este problema usaremos SVC y SVM para predecir si un determinado coche 
% tiene alta o baja autonom´ıa en la base de datos Auto. 

% Cargamos base de datos


disp('%%%%%%%%%%%%%%%%% EJERCICIO 2 %%%%%%%%%%%%%%%%%');
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
% Apartado 2 - Ajustar un SVC a los datos con varios valores de C con el 
% objetivo de predecir si un coche tiene alta o baja autonom´ıa. Reportar 
% el error CV asociado a los diferentes valores del parámetro C. 
% Comentar los resultados.
rng(1); % Fijamos semilla para el generado de números aleatorios
hpartition = cvpartition(length(y),'Holdout',0.50); % Partición no estratificada
% 50% train y 50% test
% ¡OJO! -> NECESIDAD DE DEFINIR PREVIAMENTE 'y'


% Usamos 10-FOLD CV para buscar el C óptimo
rng(2)
k = 10;
c = cvpartition(length(y1),'KFold',k);% ¡OJO! -> NECESIDAD DE DEFINIR PREVIAMENTE 'y1'






fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Repetir el apartado anterior, pero esta vez usandos SVMs con 
% kernel radial con diferentes valores de gamma

% Usando las mismas particiones de 10-FOLD CV

