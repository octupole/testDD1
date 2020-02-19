/*
 * MPI.h
 *
 *  Created on: May 24, 2019
 *      Author: marchi
 */

#ifndef PARALLEL_MPI_H_
#define PARALLEL_MPI_H_
#include <vector>
#include <iostream>
#include "../Tools/MyUtilClass.h"
#include <map>
#include "MPIconfig.hpp"
#include <mpi.h>

using std::vector;
using std::cout;using std::endl;

using namespace DVECT;
using namespace MATRIX;
using Dvect=DVECT::DDvect<double>;

namespace Parallel {
class MPI {
	MPI_Comm myWorld;
	MPI_Comm myCartComm;
	vector<int> nbrs;
	size_t N_total,N_actual;
	vector<int> nc{0,0,0};
	void setDims(int, double, double, double, double);
	vector<int> findSize3D(const int, vector<int>);
	vector<int> findSize3D(const int);
	void Init();
	MPI_Comm & ResizeWorld(MPI_Comm &);
	auto _size=[](MPI_Comm & x){int size{0};MPI_Comm_size(x,&size);return size;};
	auto _rank=[](MPI_Comm & x){int rank{-1};MPI_Comm_rank(x,&rank);return rank;};
public:
	MPI();
	MPI(double,double,double,double);
	void CartInit();


	template <typename T>
	void CartSend(Cartesian[2], vector<T> &, vector<T> &);

	MPI_Comm & gWorld(){return myWorld;}
	void PrintInfo();
	virtual ~MPI();
};

} /* namespace System */

#endif /* PARALLEL_MPI_H_ */
