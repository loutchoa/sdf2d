function fmm2d(P,sources,nbr_sources,h)
% Execute deep eikonal solver algorithm
% Parameters:
%   P : weight domain 2D double array
%   sources : 1D array as {x1,y1,x2,y2,...,xn,yn} (matlab coordinates)
%   nbr_sources : the nomber of points sources
%   h : distance between 2 adjacent points on the grid
%
%   This file is the m-file interface to the C++ fmm2d mex file.
