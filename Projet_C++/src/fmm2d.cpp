#include <mex.h>
#include "minHeap.h"
#include <cmath>
#include <vector>
#include "torch\script.h"

#define MIN(a,b) ((a)>(b)?(b):(a))

// Return the approximated distance of the point
double deep_solver(std::vector<double> patch, torch::jit::script::Module local_solver) {

    double distance, maxi, mini;

	maxi = 0;
	mini = HUGE_VAL;
	for (unsigned k = 1; k < patch.size(); k++) {
		if (mini > patch[k]) {
			mini = patch[k];
		}
		if (maxi < patch[k]) {
			maxi = patch[k];
		}
	}

	patch[0] = patch[0] / (maxi - mini);
	for (unsigned k = 1; k < patch.size(); k++) {
		patch[k] = (patch[k] - mini) / (maxi - mini);
	}

	double* x = patch.data();

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back((torch::from_blob(x, { 1, 13 }, torch::kFloat64)).toType(at::kFloat));

	torch::Tensor output = local_solver.forward(inputs).toTensor();
	distance = output[0].item<double>();
	distance = distance * (maxi - mini) + mini;
	return distance;
	
	// Local numeric solver
	/*
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
	*/
}

void fmm2d(double* D, double* P, double* sources,double h, int m, int n, int sources_nbr, char* path_model) {

	// All set to false
	bool* visited = (bool*)mxCalloc(m * n, sizeof(bool));

	double distance, obj;

	int x, y, ind, i, j, indNeigh, ip, jp;

	// Index of neighboors and patch for solver
	int ineigh[] = { -1, 1, 0, 0 };
	int jneigh[] = { 0, 0, -1, 1 };
	// Same order than the network inputs
	int ipatch[] = { 0, 0, 1,-1, -1, 1, 1,-1, 0, 0, 2,-2 };
	int jpatch[] = { 1,-1, 0, 0, -1, 1,-1, 1, 2,-2, 0, 0 };
	std::vector<double> dist_patch;
    
	// Creation of minHeap
	minHeap wavefront = minHeap(m * n);


	// Load the model
	std::string p(path_model);
	torch::jit::script::Module local_solver;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		local_solver = torch::jit::load(p);
	}
	catch (const c10::Error& e) {
		std::cout << e.msg() << std::endl;
		std::cout << "error loading the model\n";
	}

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

					obj = HUGE_VAL;
					for (unsigned s = 0; s < sources_nbr; s++) {
						obj = MIN(sqrt(pow(i - sources[2 * s], 2) + pow(j - sources[2 * s + 1], 2)), obj);
					}

					if ((ip >= 1) && (jp >= 1) && (ip <= m) && (jp <= n) && (D[(ip - 1) + (jp - 1) * m] != HUGE_VAL) && (D[(ip - 1) + (jp - 1) * m] < obj)) {
							dist_patch.push_back(D[(ip - 1) + (jp - 1) * m]);
					}
					else {
						// distance for HUGE_VAL points = ground truth distance + 2step
						dist_patch.push_back(obj+2*h);
                    }
					
				}

				distance = deep_solver(dist_patch, local_solver);

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
    mxFree(visited);
}

void mexFunction(int nlhs, mxArray* plhs[],	int nrhs, const mxArray* prhs[]) {


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

	char* buf;
	int   buflen;

	/* Find out how long the input string array is. */
	buflen = ((int)mxGetM(prhs[4]) * (int)mxGetN(prhs[4])) + 1;

	/* Allocate enough memory to hold the converted string. */
	buf = (char* )mxCalloc(buflen, sizeof(mxChar));
	if (buf == NULL) mexErrMsgTxt("Not enough heap space to hold converted string.");

	/* Copy the string data from prhs[0] and place it into buf. */
	mxGetString(prhs[4], buf, buflen);

	// Create output matrix and its data pointer
	plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
	D = mxGetPr(plhs[0]);
	
	fmm2d(D, P, sources, h, m, n, sources_nbr, buf);
}