#include <mex.h>
#include "minHeap.h"
#include <cmath>
#include <vector>

#define MIN(a,b) ((a)>(b)?(b):(a))

// Return the approximated distance of the point
double deep_solver(std::vector<double> patch) {
    double t1 = MIN(patch[1],patch[2]);
    double t2 = MIN(patch[3],patch[4]);
    double t3 = MIN(patch[5],patch[6]);
    double t4 = MIN(patch[7],patch[8]);
    double d,d_diag;
    
    if ((abs(t1-t2) < patch[0]) && (t1 != HUGE_VAL) && (t2 != HUGE_VAL)) {
        d = (t1+t2+sqrt(2*pow(patch[0],2)-pow(t2-t1,2)))/2;
    }
    else {
        d = MIN(t1,t2) + patch[0];
    }
    if ((abs(t3-t4) < sqrt(2)*patch[0]) && (t3 != HUGE_VAL) && (t4 != HUGE_VAL)) {
        d_diag = (t3+t4+sqrt(4*pow(patch[0],2)-pow(t4-t3,2)))/2;
    }
    else {
        d_diag = MIN(t3,t4) + sqrt(2)*patch[0];
    }
    return MIN(d,d_diag);
}

void fmm2d(double* D, double* P, double* sources,double h, int m, int n, int sources_nbr) {

	// All set to false
	bool* visited = (bool*)mxCalloc(m * n, sizeof(bool));

	double distance;

	int x, y, ind, i, j, indNeigh, ip, jp;

	// Index of neighboors and patch for solver
	int ineigh[] = { -1, 1, 0, 0 };
	int jneigh[] = { 0, 0, -1, 1 };
	// Same order than the network inputs
	int ipatch[] = { 0, 0, 1,-1, -1, 1, 1,-1, 0, 0, 2,-2 };
	int jpatch[] = { 1,-1, 0, 0, -1, 1,-1, 1, 2,-2, 0, 0 };
	std::vector<double> dist_patch;
    
	minHeap wavefront = minHeap(m * n);

	// Initialization of distance matrix
	for (int k = 0; k < m*n; k++) D[k] = HUGE_VAL;
    
	// Initialization of wavefront : put sources points in it
	for (int k = 0; k < sources_nbr; k++) {
		x = sources[2 * k];
		y = sources[2 * k + 1];
		ind = (x-1) + (y-1)*m;
		D[ind] = 0;
		wavefront.insert(ind, 0);
	}
    
	while (wavefront.size() != 0) {

		// get top of the queue
		ind = wavefront.extractMin();
		x = ind + 1 - (ind / m) * m;
		y = (ind / m) + 1;

		// Tag as visited
		visited[ind] = true;
        
		// For each adjacent point not already visited:
		for (int k = 0; k < 4; k++) {
			i = x + ineigh[k];
			j = y + jneigh[k];
			indNeigh = (i - 1) + (j - 1) * m;
            
			if ((i >= 1) && (j >= 1) && (i <= m) && (j <= n) && !visited[indNeigh]) {

				// Construct the patch 
				dist_patch.clear();
				dist_patch.push_back(h);
				for (int p = 0; p < 12; p++) {
					ip = i + ipatch[p];
					jp = j + jpatch[p];
					if ((ip >= 1) && (jp >= 1) && (ip <= m) && (jp <= n) && D[(ip - 1) + (jp - 1) * m] != HUGE_VAL) {
						dist_patch.push_back(D[(ip - 1) + (jp - 1) * m]);
					}
					else {
						// distance for HUGE_VAL points = ground truth distance + 2step
						distance = HUGE_VAL;
						for (unsigned s = 0; s < sources_nbr; s++) {
							distance = MIN(sqrt(pow(i - sources[2 * s], 2) + pow(j - sources[2 * s + 1], 2))+2*h, distance);
						}
						dist_patch.push_back(distance);
                    }
				}

				distance = deep_solver(dist_patch);

				if (wavefront.isInHeap(indNeigh)) {
					if (distance < D[indNeigh]) {
						wavefront.decrease(indNeigh, distance);
						D[indNeigh] = distance;
					}
				}
				else {
					D[indNeigh] = distance;
					wavefront.insert(indNeigh, distance);
				}
			}
		}
	}
    
	//wavefront.~minHeap();
    mxFree(visited);
}

void mexFunction(int nlhs, mxArray* plhs[],
	int nrhs, const mxArray* prhs[]) {


	int m, n, sources_nbr;
	double h;
	double* P, * D, * sources;

	// P : the weighed matrix, sources : sources points
	// UNUSED FOR NOW
	P = mxGetPr(prhs[0]);
	sources = mxGetPr(prhs[1]);

	// grid m,n 
	m = (int)mxGetM(prhs[0]);
	n = (int)mxGetN(prhs[0]);
	sources_nbr = (int)mxGetScalar(prhs[2]);

	// Get the step h
	h = (double)mxGetScalar(prhs[3]);

	// Create output matrix and its data pointer
	plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
	D = mxGetPr(plhs[0]);
	
	fmm2d(D, P, sources, h, m, n, sources_nbr);
}