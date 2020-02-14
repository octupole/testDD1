/*
 * MPI.h
 *
 *  Created on: May 24, 2019
 *      Author: marchi
 */

#ifndef PARALLEL_MPI_H_
#define PARALLEL_MPI_H_
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include  <boost/mpi/group.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>
#include <iostream>
#include "../Tools/MyUtilClass.h"
#include <map>
#include "MPIconfig.hpp"
using std::vector;
using std::cout;using std::endl;
namespace mpi = boost::mpi;

using namespace DVECT;
using namespace MATRIX;
using Dvect=DVECT::DDvect<double>;

namespace Parallel {
class MPI {
	mpi::communicator world0;
	mpi::communicator world;
	mpi::cartesian_communicator * Cartx{nullptr};
	mpi::cartesian_dimension Dims[3];
	mpi::environment env;
	MPI_Comm myWorld;
	MPI_Comm myCartComm;
	vector<int> nbrs;
	size_t N_total,N_actual;
	vector<int> nc{0,0,0};
	void setDims(int, double, double, double, double);
	vector<int> findSize3D(const int, vector<int>);
	vector<int> findSize3D(const int);
	void Init();
	mpi::communicator ResizeWorld(mpi::communicator);
public:
	MPI();
	MPI(double,double,double,double);
	void CartInit();


	template <typename T>
	void CartSend(Cartesian[2], vector<T> &, vector<T> &);

	mpi::communicator & gWorld(){return world;}
	void PrintInfo();
	virtual ~MPI();
};

} /* namespace System */

#endif /* PARALLEL_MPI_H_ */
