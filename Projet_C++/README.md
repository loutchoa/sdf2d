# ProjetYohan C++ 

0) Download libtorch (without cuda and release mod) and unzip it where you want (you'll need the path) 

1) Modify the line in CMakeLists.txt with your path line 9

2) In your project's root :
	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=[PATH TO LIBTORCH] .. #Libtorch without cuda
	make  # on unix (didn't test but should work here)
	cmake --build . -config Release # on windows, use release because it's debug by default

3) Then open your project folder on matlab and make sure to have everything in the path (the mex file should be in the build/Release)

4) Mex already created so you just have to call for the function :
 
deep_eikonal_solver_c(P,sources,nbr_sources,h)
% Execute deep eikonal solver algorithm (return distance matrix)
% Parameters:
%   P : weight domain 2D double array
%   sources : 1D array as {x1,y1,x2,y2,...,xn,yn} (matlab coordinates)
%   nbr_sources : the nomber of points sources
%   h : distance between 2 adjacent points on the grid