#include <stdlib.h>
#include <math.h>
#include <mex.h>
#include "minHeap.h"
#include <stdio.h>
#include <limits>
#include <iostream>
using namespace std;

minHeap::minHeap(int N) {

	// Estimation of minHeap size
	max_size = N / 4 + 1;

	heapSize = 0;

	// Initialize Heap and H2P which are the same size.
	Keys = (double*)mxCalloc(max_size, sizeof(double));
	H2P = (int*)mxCalloc(max_size, sizeof(int));

	P2H = (int*)mxCalloc(N, sizeof(int));
	// Initialize P2H by setting all values to -1, signifying that
	// there are no heap elements corresponding to the pixels.
	for (int i = 0; i < N; i++) {
		P2H[i] = -1;
	}
}

minHeap::~minHeap() {
	// Destructor
	mxFree(Keys);
	mxFree(H2P);
	mxFree(P2H);
}

bool minHeap::isInHeap(int i) {
	return (P2H[i] >= 0);
}

int minHeap::size() {
	return heapSize;
}

void minHeap::insert(int i, double distance) {

	Keys[heapSize] = HUGE_VAL;
	H2P[heapSize] = i;
	P2H[i] = heapSize;

	heapSize++;

	decrease(i, distance);
}

void minHeap::decrease(int i, double distance) {

	int ind = P2H[i];
	Keys[ind] = distance;

	while (ind > 0 && Keys[ind] < Keys[parent(ind)]) {
		swap(ind, parent(ind));
		ind = parent(ind);
	}
}

int minHeap::extractMin() {

	// Give the index of the point corresponding to this min
	int point = H2P[0];

	swap(0, heapSize - 1);
	heapSize--;
	heapify(0);

	return point;
}

void minHeap::print() {
		mexPrintf("\nminHeap:  ");
		for(int k=0; k<heapSize; k++)
				mexPrintf("%.3g ",Keys[k]);
}

void minHeap::heapify(int i) {

	int l, r, smallest;
	while (true) {

		l = left(i);
		r = right(i);
		smallest = i;

		if (l < heapSize && Keys[l] < Keys[i]) smallest = l;
		if (r < heapSize && Keys[r] < Keys[smallest]) smallest = r;
		if (smallest == i) break;
		swap(i, smallest);
		i = smallest;
	}
}

void minHeap::swap(int i1, int i2) {
	double tmpKey;
	int tmpInd;

	if (i1 == i2)
		return;

	// Swap keys
	tmpKey = Keys[i1];
	Keys[i1] = Keys[i2];
	Keys[i2] = tmpKey;

	// Swap P2H values
	P2H[H2P[i1]] = i2;
	P2H[H2P[i2]] = i1;

	// Swap H2P elems
	tmpInd = H2P[i1];
	H2P[i1] = H2P[i2];
	H2P[i2] = tmpInd;
}