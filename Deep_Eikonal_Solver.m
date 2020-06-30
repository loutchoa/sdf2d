%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         Deep Eikonal Solver                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear;

% Dimension de l'espace
n = 200;

% L'utilisateur choisit le nbre de points de départ
nb_points = 0;
while (nb_points <1) || (nb_points >= n)
    nb_points = input( 'Combien de points de départs ? ' );
end

% Affichage de la carte de potentiel
P = matrice_poids('constant', n);
figure(2);
imagesc(P); axis image; axis off;colormap gray(256);

% Choix point de départ
disp('Point(s) de départ:');
points = 0;
ind_s = [];
while points < nb_points
    hold on;
    nouveau = round(ginput(1));
    % ind_s = [2, 15]; % pour la position du rapport en mode binaire
    ind_s = [ind_s; nouveau(2), nouveau(1)];
    plot( nouveau(1), nouveau(2), 'rx' );
    points = points + 1;
end
hold off;

%% Initialisation algorithme

% Gère l'affichage
verbose = 1;

% Les trois états possibles d'un point
visited = -1;
wavefront = 0;
not_visited = 1;

% Matrice des distances initialisée à inf
D = P.*0 + inf;
D(sub2ind(size(P), ind_s(:,1), ind_s(:,2))) = 0;

% Ensemble de de départ, les sommets sont triés colonne par colonne
% Sommet en (i,j) = n*(j-1)+i (ou sub2ind), matlab (indice commencent à 1)
S = ones(n);
S(sub2ind(size(P), ind_s(:,1), ind_s(:,2))) = visited;

% WaveFront
WV = MinHeap(min(n^2,2^48));

% Newly visited points
NVP = ind_s;

str = 'Performing Fast Marching algorithm.';
if verbose
    b = waitbar(0,str);
end

sommets_visites = size(ind_s,1);
nb_iter_max = n^2+1; 
nbvoisins = 8;
iter = 0;
while (iter<nb_iter_max) && (sommets_visites ~= n^2)
    iter = iter + 1;
    
    % "Denote the points adjacent to the newly visited points as A"
    A = points_adjacents(NVP,n,"gestion bord",nbvoisins);
    
    % On ne garde que les sommets qui n'ont pas encore été visités
    Non_visite = S(sub2ind(size(P), A(:,1), A(:,2))) ~= visited;
    A = A([Non_visite Non_visite]);
    A = reshape(A,length(A)/2,2);
    
    % "Estimate u(p) for p in A based on visited points"
    D = Local_Numerical_Solver(A,D,P,nbvoisins);
    
    % "Tag each p in A as wavefront"
    for i=1:size(A,1)
        
        % On stocke des key-value dans le tas binaire
        % key = distance
        kv.key = D(A(i,1), A(i,2));
        % value = coordonnées
        kv.value = A(i,:);
        
        % Si c'est un nouveau point, on l'insert dans le wavefront
        if S(sub2ind(size(P), A(i,1), A(i,2))) == not_visited
            WV.InsertKey(kv);
        % Sinon on le met à jour avec sa nouvelle distance
        else
            WV.Update(kv);
        end
    end
    S(sub2ind(size(P), A(:,1), A(:,2))) = wavefront;
    
    % "Tag the least distant wavefront point as visited"
    % On choisit le point du front d'onde de distance minimal
    min = WV.ExtractMin();
    
    % Point choisi
    i = min.value(1);
    j = min.value(2);
    S(i,j) = visited; 
    sommets_visites = sommets_visites + 1;
    
    % Ajout du nouveau point visité à la liste (qui etait alors vide)
    NVP = [i j];
    
    if verbose
        waitbar(iter/nb_iter_max, b, sprintf('Performing Fast Marching algorithm, iteration %d.', iter) );
    end
end
figure(3);
subplot(1,2,1)
imagesc(D); axis image; axis off;%colormap default;
hold on;
plot( ind_s(:,2), ind_s(:,1), 'rx' );
hold off;

subplot(1,2,2)
imagesc(S); axis image; axis off;colormap gray(256);
hold on;
plot( ind_s(:,2), ind_s(:,1), 'rx' );
hold off;

figure(4);
x = 0:1/(n-1):1;
[X,Y] = meshgrid(x,x);
contourf(X,1-Y,D,15);
hold on;
pts_x = X(ind_s(:,1),ind_s(:,2));
pts_x = pts_x(1,:);
pts_y = Y(ind_s(:,1),ind_s(:,2));
pts_y = pts_y(:,1);
plot( pts_x, 1-pts_y, 'rx' );
hold off;
daspect([1 1 1]);

% Le réseau de neurones
layers = [
    sequenceInputLayer(13,"Name","sequence")
    fullyConnectedLayer(128,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(256,"Name","fc_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(128,"Name","fc_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(1,"Name","fc_4")
    reluLayer("Name","relu_4")
    regressionLayer("Name","regressionoutput")];

% Affichage du réseau de neurones
%plot(layerGraph(layers));