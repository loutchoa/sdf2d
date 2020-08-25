%% Remember to add build/Release to the path ! 

m = 100;
n = 100;

step = 1; % if step > 1, use the right solver ! 

% Works with step from 1 to 2 but less accurate (need more data)
% path_solver = 'C:\Users\yohan\Desktop\Cours\2A\Stage\ProjetYohan\Projet_C++\local_solver_step1-2.pt'; 

% step = 1, more accurate because all data trained for step = 1
path_solver = 'C:\Users\yohan\Desktop\Cours\2A\Stage\ProjetYohan\Projet_C++\local_solver.pt'; 

% % [x0 y0 x1 y1 ... x y]
sources = [ 30 50 70 50];

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
     gt(:,:,i) = sqrt((step*(Y-xi)).^2 + (step*(X-yi)).^2);
end
gt = min(gt,[],3);

figure();
contourf(X,Y,des,10);
hold on;
pts_x = X(sources(1:2:end),sources(2:2:end));
pts_y = Y(sources(1:2:end),sources(2:2:end));
plot( pts_x(1,:), pts_y(:,1), 'rx');
hold off;
daspect([m n 1]);
