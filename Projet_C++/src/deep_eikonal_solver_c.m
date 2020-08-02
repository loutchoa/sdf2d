function deep_eikonal_solver_c(P,sources,nbr_sources,h,p)
% Execute deep eikonal solver algorithm
% Parameters:
%   P : weight domain 2D double array
%   sources : 1D array as {x1,y1,x2,y2,...,xn,yn} (matlab coordinates)
%   nbr_sources : the nomber of points sources
%   h : distance between 2 adjacent points on the grid
%   p : path to the saved model : 'path/to/model/local_solver.pt'
%
%   This file is the m-file interface to the C++ fmm2d mex file.
