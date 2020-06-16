function A = points_adjacents4(liste_points,n,option)
% Renvoie la liste des points adjacents(4 adjacents) à ceux passés en arguments
%
%   A = points_adjacents4(liste_points,n)
%
%   'liste_points' la liste des points en colonne
%   'n' la dimension de l'espace
%   'option' permet de différencier le cas où on veut renvoyer les 4 
%            coordonnées voisines même si elles n'appartiennent pas à 
%            l'espace (= "avec") et le cas où on veut renvoyer les vrais voisins 
%            en considérant les bords (=autres) 

x = liste_points(:,1);
y = liste_points(:,2);
A = [x+1 y; x-1 y; x y+1; x y-1];
if option ~= "avec"
    A = A.*(A>0 & A<=n);
    A = A(all(A,2),:);
end
end