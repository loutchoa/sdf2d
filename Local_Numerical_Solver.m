function D = Local_Numerical_Solver(A,D,P)
%   Approximation locale du gradient
%
%   D = Local_Numerical_Solver(A,D,P)
%
%   'A' la liste des points dont on doit calculer la distance
%   'D' la matrice des distances
%   'P' la matrice des poids

% Dimension de l'espace
n = length(D);

% Pas de discrétisation 
h = 1/(n-1);

for p = 1:size(A,1)
    i = A(p,1);
    j = A(p,2);
    
    voisins = points_adjacents4(A(p,:),n,"avec");
    % t1 = min{u_i-1,j ,u_i+1,j}
    t1 = min(distance(D,n,voisins(1:2,:)));
    % t2 = min{u_i,j-1 ,u_i,j+1}
    t2 = min(distance(D,n,voisins(3:4,:)));
    
    % Upwind resolution avec un pas de régularistion pondéré par la carte
    % de potentiel
    h_pondere = h/P(i,j);
    
    if abs(t1-t2) < h_pondere 
        u_p = (t1+t2+sqrt(2*h_pondere^2-(t2-t1)^2))/2;
    else
        u_p = min(t1,t2) + h_pondere;
    end
    if D(i,j) > u_p
       D(i,j) = u_p; 
    end
end
end

% Permet de retourner les distances à partir d'un ensemble de points
% On renvoit inf pour un point qui n'est pas dans la grille
function d = distance(D,n,points)
dans_grille = points.*(points>0 & points<=n);
dans_grille = dans_grille(all(dans_grille,2),:);

d = [D(sub2ind(size(D), dans_grille(:,1), dans_grille(:,2)))];
d = [d ; ones(size(points,1)-size(dans_grille,1),1)*inf];
end