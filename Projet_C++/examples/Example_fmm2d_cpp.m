%% Remember to add build/Release to the path ! 

m = 100;
n = 100;

path_solver = 'C:\Users\yohan\Desktop\Cours\2A\Stage\ProjetYohan\Projet_C++\local_solver.pt';

% [x0 y0 x1 y1 ... x y]
sources = [ 20 20 20 80 80 20 80 80 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];

step = 1; % model isn't trained for other steps for now

%% Don't need to modify 
nb_sources = length(sources)/2;

C = deep_eikonal_solver_c(ones(m,n),sources,nb_sources,step,path_solver);

figure();
x = 0:1/(m-1):1;
y = 0:1/(n-1):1;
[X,Y] = meshgrid(y,x);
contourf(X,1-Y,C,10);
hold on;
pts_x = X(sources(1:2:end),sources(2:2:end));
pts_x = pts_x(1,:);
pts_y = Y(sources(1:2:end),sources(2:2:end));
pts_y = pts_y(:,1);
plot( pts_x, 1-pts_y, 'rx' );
hold off;
daspect([m n 1]);

