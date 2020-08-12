%% Remember to add build/Release to the path ! 

m = 100;
n = 100;

path_solver = 'C:\Users\yohan\Desktop\Cours\2A\Stage\ProjetYohan\Projet_C++\local_solver.pt';

% [x0 y0 x1 y1 ... x y]
sources = [ 30 40 70 60];

step = 1; % model isn't trained for other steps for now

%% Don't need to modify 
nb_sources = length(sources)/2;

des = deep_eikonal_solver_c(ones(m,n),sources,nb_sources,step,path_solver);

%Ground truth distances
x = 1:m;
y = 1:n;
[X,Y] = meshgrid(x,y); 
nsources = reshape([sources(1:2:end) sources(2:2:end)],[nb_sources,2]);
gt = zeros(m,n,nb_sources);
for i=1:nb_sources
     xi = nsources(i,1);
     yi = nsources(i,2);
     gt(:,:,i) = sqrt((Y-xi).^2 + (X-yi).^2);
end
gt = min(gt,[],3);

figure();
contour(X,Y,des,10,'r');
hold on;
contour(X,Y,gt,10,'b');
hold on;
pts_x = X(sources(1:2:end),sources(2:2:end));
pts_y = Y(sources(1:2:end),sources(2:2:end));
plot( pts_x(1,:), pts_y(:,1), 'kx');
legend('FMM Deep','Ground Truth','Sources')
hold off;
daspect([m n 1]);

% ZOOM if you want to see differences bewteen the two