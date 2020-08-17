# ProjetYohan C++ 

0) Download libtorch without cuda (release mod for windows) and unzip it where you want (you'll need the path) 

1) Modify the line in CMakeLists.txt with your path of matlabroot line 9

2) In your project's root :
	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=[PATH TO LIBTORCH] .. #Libtorch without cuda
	make  # on unix 
	cmake --build . -config Release # on windows, use release because it's debug by default

3) Then open your project folder on matlab and make sure to have everything in the path (the mex file should be in the build/Release)

4) Mex already created so you just have to call for the function (there is an example)
 