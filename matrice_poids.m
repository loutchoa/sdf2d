function P = matrice_poids(nom, n)
% matrice_poids permet de charger les données d'une carte de potentiel
%
%   P = matrice_poids(nom, n)
%
%   'P' la carte de potentiel

switch nom
    case 'gaussien'
        x = -1:2/(n-1):1;
        [Y,X] = meshgrid(x,x);
        sigma = 0.8;
        P = exp( -(X.^2+Y.^2)/sigma^2 );
        P = 1-rescale( P ) + 0.01;
    case 'constant'
        P = ones(n)*0.5;
    case 'pics'
        P = rescale( peaks(n) )+0.01;
    case 'binaire'
        P = ones(n)-0.25; 
        P(1:end/2,:) = .25;
    case 'angle'
        x = 0:1/(n-1):1;
        [X,Y] = meshgrid(x,0:1/(n-1):1);
        P = X.*Y + 0.01;
    otherwise %gradient      
        P = ones(n).*(0:1/(n-1):1)+0.01;
end