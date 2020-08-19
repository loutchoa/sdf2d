# ProjetYohan

This project's purpose is to construct an efficient deep eikonal solver.

- The first part of the project is in the projet_matlab folder. It uses a local numeric solver using a first order scheme with 4 or 8 neighbors patch.

- The second part is in both Reseau_pytorch and Projet_C++ folders. 
The first one contains all the code used to create the model (learning, database...).
The second one contains the main code of the deep eikonal solver algorithm. There is a README file in it which explains how to compile and generate the MEX file.