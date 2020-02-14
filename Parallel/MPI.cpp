/*
 * MPI.cpp
 *
 *  Created on: May 24, 2019
 *      Author: marchi
 */

#include "MPI.h"
#include <mpi.h>
#include <cmath>
#include <boost/range/irange.hpp>
#include "../Tools/MyTools.hpp"
using outputs::Log;


namespace Parallel {

vector<int> MPI::findSize3D(int n){
	int m=floor(pow(n,1/3));
	vector<int> result{m,m,m};

	if(m*m*m == n){
		return result;
	} else{
		while(true){
			for( size_t o{0};o<DIM;o++){
				result[o]++;
				if(result[XX]*result[YY]*result[ZZ] == n) return result;
				if(result[XX]*result[YY]*result[ZZ] > n) {
					result[o]--;
					return result;
				}

			}
		}
	}
};

vector<int> MPI::findSize3D(const int n,vector<int> vn){
	vector<int> result(vn);
	while(true){
		for( size_t o{0};o<DIM;o++){
			result[o]--;
			if(result[XX]*result[YY]*result[ZZ] <= n) return result;
		}
	}
};

void MPI::setDims(const int n, double cut, double a, double b, double c){
	vector<int> my{static_cast<int>(a/cut),static_cast<int>(b/cut),static_cast<int>(c/cut)};
	if(my[XX]*my[YY]*my[ZZ] == n){
		nc=my;
	}else if(my[XX]*my[YY]*my[ZZ] < n){
		int n_old=n;
		nc=my;
	}else if(my[XX]*my[YY]*my[ZZ] > n){
		nc=this->findSize3D(n,my);
	}
}
mpi::communicator MPI::ResizeWorld(mpi::communicator worldx0){
	if(world0.size() != nc[XX]*nc[YY]*nc[ZZ]){
		mpi::group local = worldx0.group();
		boost::integer_range<int> r = boost::irange(nc[XX]*nc[YY]*nc[ZZ]-1, worldx0.size()-1);
		mpi::group subgroup = local.exclude(r.begin(), r.end());
		mpi::communicator others{worldx0, subgroup};
		return others;
	} else{
		return world0;
	}

}

void MPI::PrintInfo(){
	if(world0.size() != world.size()){
		Log() << " MPI size has been changed for Cartesian decomposition." ;
		Log() << "\tOld size: "<< N_total;
		Log() << "\tNew size: "<< N_actual<<": " << nc[XX]<< " X " << nc[YY]<< "  X " << nc[ZZ]<< " " ;

	} else{
		Log() << " MPI size for Cartesian decomposition." ;
		Log() << "\tMPI size: "<< N_total<<": " << nc[XX]<< " X " << nc[YY]<< "  X " << nc[ZZ]<< " " ;
	}
}
//template <typename T>
//void MPI::CartSend(std::map<int,T> & m){
//	std::vector<T> outbuf(CARTDIRS);
//	std::vector<T> inbuf(CARTDIRS);
//	int tag{1};
//	for(size_t o{0};o<CARTDIRS;o++){
//		outbuf[o]=m[o];
//	}
//	mpi::request reqs[12];
//	for(size_t o{0};o<CARTDIRS;o++){
//		int dest{nbrs[o]},source{nbrs[o]};
//		reqs[o]=Cartx->isend(dest,tag,outbuf[o]);
//		reqs[o+CARTDIRS]=Cartx->irecv(source,tag,inbuf[o]);
//	}
//	mpi::wait_all(reqs,reqs+2*CARTDIRS);
//	printf("rank= %d inbuf(u,d,l,r,n,s)= %d %d %d %d %d %d\n",
//		             Cartx->rank()
//					 ,inbuf[UP],inbuf[DOWN],inbuf[LEFT],inbuf[RIGHT],inbuf[NORTH],inbuf[SOUTH]);
//
//}
template <typename T>
void MPI::CartSend(Cartesian dir[2], vector<T> & one, vector<T> & two){
	int tag{1};
	mpi::request reqs[4];

	int o0 =static_cast<int> (dir[0]);
	int o1=static_cast<int> (dir[1]);
	int dest[2]{nbrs[o0],nbrs[01]},source[2]{nbrs[o0],nbrs[o1]};

	reqs[0]=Cartx->isend(dest[0],tag,one.size());
	reqs[1]=Cartx->isend(dest[1],tag,two.size());
	size_t nOne,nTwo;
	reqs[2]=Cartx->irecv(source[0],tag,nOne);
	reqs[3]=Cartx->irecv(source[1],tag,nTwo);
	mpi::wait_all(reqs,reqs+4);
	vector<T> inrcv0=vector<T>(nOne);
	vector<T> inrcv1=vector<T>(nTwo);
	reqs[0]=Cartx->isend(dest[0],tag,one);
	reqs[1]=Cartx->isend(dest[1],tag,two);
	reqs[2]=Cartx->isend(source[0],tag,inrcv0);
	reqs[3]=Cartx->isend(source[1],tag,inrcv1);
	mpi::wait_all(reqs,reqs+4);

}

//void MPI::CartSend(vector<int> outbuf){
//	for(int o{0};o<6;o++){
//		if(o%2 == 0) outbuf[o]=Cartx->rank();
//		else outbuf[o]=-Cartx->rank();
//	}
//	int inbuf[6],tag{1};
//	mpi::request reqs[12];
//	for(size_t o{0};o<6;o++){
//		int dest{nbrs[o]},source{nbrs[o]};
//		reqs[o]=Cartx->isend(dest,tag,outbuf[o]);
//		reqs[o+6]=Cartx->irecv(source,tag,inbuf[o]);
//	}
//	mpi::wait_all(reqs,reqs+12);
//
//
//	printf("rank= %d inbuf(u,d,l,r,n,s)= %d %d %d %d %d %d\n",
//		             Cartx->rank()
//					 ,inbuf[UP],inbuf[DOWN],inbuf[LEFT],inbuf[RIGHT],inbuf[NORTH],inbuf[SOUTH]);
//
//}
void MPI::CartInit(){
	const int LA{static_cast<int>(Cartesian::north)};
	int g=LA;
	Dims[XX]={nc[XX], true};
	Dims[YY]={nc[YY], true};
	Dims[ZZ]={nc[ZZ], true};
	Cartx=new mpi::cartesian_communicator(world, mpi::cartesian_topology(Dims));
	this->myCartComm=(*Cartx);
	MPI_Cart_shift(myCartComm,0,1,&nbrs[UP],&nbrs[DOWN]);
	MPI_Cart_shift(myCartComm,1,1,&nbrs[LEFT],&nbrs[RIGHT]);
	MPI_Cart_shift(myCartComm,2,1,&nbrs[NORTH],&nbrs[SOUTH]);

//	printf("rank= %d coords= %d %d %d  neighbors(u,d,l,r,n,s)= %d %d %d %d %d %d\n",
//	             Cartx->rank()
//				 ,Cartx->coordinates(Cartx->rank())[0]
//				 ,Cartx->coordinates(Cartx->rank())[1]
//				 ,Cartx->coordinates(Cartx->rank())[2]
//				 ,nbrs[UP],nbrs[DOWN],nbrs[LEFT],nbrs[RIGHT],nbrs[NORTH],nbrs[SOUTH]);
}
void MPI::Init(){
	N_total=world0.size();
	world=this->ResizeWorld(world0);
	if((world) == MPI_COMM_NULL){
		   MPI_Finalize();
		   exit(0);
	}else{
		myWorld=(world);
	}
	N_actual=world.size();
	nbrs=vector<int>(N_actual);
	Log() << "MPI Cartesian grid: "<<nc[XX]<< " "<< nc[YY]<< " "<< nc[ZZ]<< " ";
	Log() << "MPI size: "<<N_actual ;
}
MPI::MPI(double cut, double a, double b, double c){

	if(world0.rank() != 0)
		std::cout.setstate(std::ios::failbit);

	this->setDims(world0.size(),cut,a,b,c);
	this->Init();
}
MPI::MPI(){
	if(world0.rank() != 0)
		std::cout.setstate(std::ios::failbit);

	nc=this->findSize3D(world0.size());
	this->Init();
}

MPI::~MPI() {
	MPI_Finalize();
}

template void MPI::CartSend<double>(Cartesian[2], vector<double> &, vector<double> &);
} /* namespace System */
