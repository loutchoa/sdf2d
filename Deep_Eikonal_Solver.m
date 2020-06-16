%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         Deep Eikonal Solver                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear;

% Dimension de l'espace
n = 50;

% Affichage de la carte de potentiel
P = matrice_poids('binaire', n);
imagesc(P); axis image; axis off;colormap gray(256);
hold on;

% Choix point de départ
disp('Point de départ:');
ind_s = round(ginput(1));
ind_s = [ind_s(2), ind_s(1)];
disp(ind_s)
plot( ind_s(:,2), ind_s(:,1), 'rx' );
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
WV = [];

% Newly visited points
NVP = ind_s;

str = 'Performing Fast Marching algorithm.';
if verbose
    b = waitbar(0,str);
end

sommets_visites = size(ind_s,1);
nb_iter_max = n^2+1;
iter = 0;
figure(5);
while (iter<nb_iter_max) && (sommets_visites ~= n^2)
    iter = iter + 1;
    
    % "Denote the points adjacent to the newly visited points as A"
    A = points_adjacents4(NVP,n,"gestion bord");
    % On ne garde que les sommets qui n'ont pas encore été visités
    Non_visite = S(sub2ind(size(P), A(:,1), A(:,2))) ~= visited;
    A = A([Non_visite Non_visite]);
    A = reshape(A,length(A)/2,2);
    
    % "Estimate u(p) for p in A based on visited points"
    D = Local_Numerical_Solver(A,D,P);
    
    % "Tag each p in A as wavefront"
    S(sub2ind(size(P), A(:,1), A(:,2))) = wavefront;
    WV = [WV; A];    
    % On enlève les doublons dans le WV
    WV = unique(WV,'rows','stable');
    
    % "Tag the least distant wavefront point as visited"
    % On choisit le point du front d'onde de distance minimal
    ind_WV = sub2ind(size(P), WV(:,1), WV(:,2));
    [~,ind] = min(D(ind_WV));
    
    % selected vertex
    i = WV(ind,1);
    j = WV(ind,2);
    WV(ind,:) = [];  % pop
    S(i,j) = visited; 
    sommets_visites = sommets_visites + 1;
    
    % Ajout du nouveau point visité à la liste (qui etait alors vide)
    NVP = [i j];
    
    if verbose
        waitbar(iter/nb_iter_max, b, sprintf('Performing Fast Marching algorithm, iteration %d.', iter) );
    end
    subplot(1,2,1)
    imagesc(D); axis image; axis off;%colormap default;
    hold on;
    plot( ind_s(:,2), ind_s(:,1), 'rx' );
    hold off;

    subplot(1,2,2)
    imagesc(S); axis image; axis off;colormap gray(256);
end

figure();
x = 0:1/(n-1):1;
[X,Y] = meshgrid(x,x);
contourf(X,1-Y,D,15);

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