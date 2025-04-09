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

% Cargamos base de datos


disp('%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Divide los datos de manera aleatoria en conjuntos de 
% entrenamiento (50%) y de test (50%). Fijar la semilla para la 
% generación de números pseudo-aleatorios a rng(4).
rng(4); % Fijamos semilla para el generado de números aleatorios



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un árbol de regresión usando el conjunto de 
% entrenamiento. Visualiza el árbol e interpreta los resultados. 
% ¿Cuál es el MSE de test obtenido?
% Entrenamos árbol de regresión



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Utiliza CV para determinar el nivel óptimo de complejidad 
% del árbol. ¿Mejora el MSE de test con la poda del árbol?
rng(2);



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Utiliza bagging para analizar los datos. ¿Cuál es el MSE de 
% test obtenido?  Determina qué variables son las más importantes.
rng(4);



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Utiliza random forests para analizar los datos. ¿Cuál es el 
% MSE de test obtenido?  Determina qué variables son las más importantes. 
% Describe el efecto de m, el número de predictores considerado en cada 
% división, en la tasa de error obtenida.
rng(4);


% Analizamos el efecto de m usando CV
rng(4);
