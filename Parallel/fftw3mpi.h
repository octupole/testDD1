/*
 * fftw3mpi.h
 *
 *  Created on: Feb 11, 2020
 *      Author: marchi
 */

#ifndef PARALLEL_FFTW3MPI_H_
#define PARALLEL_FFTW3MPI_H_
#include <pfft.h>
#include <complex>
#include <iostream>
#include <string>
#include <vector>
#include "../Tools/Ftypedefs.h"
#include "../Tools/Array.h"
using Complex=std::complex<double>;
using namespace Array;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace Parallel {
auto findSize3D=[](int n)->vector<int>{
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

class fftw3mpi {
protected:
	MPI_Comm * comm_cart_3dx{nullptr};
	ptrdiff_t np[DIM]{-1,-1,-1};
	ptrdiff_t NN[DIM]{-1,-1,-1};
	ptrdiff_t local_ni[3]{0,0,0}, local_i_start[3]{0,0,0};
	ptrdiff_t local_no[3]{0,0,0}, local_o_start[3]{0,0,0};
	array1<double> in;
	array1<Complex> out;
	void _copy_to_in(array3<double> &);
	void _copy_from_in(array3<double> &);
	void _copy_to_out(array3<Complex> &);
	void _copy_from_out(array3<Complex> &);
public:
	fftw3mpi(int, int, int);
	fftw3mpi();
	bool isInitialized();
	void setDimFFT(int,int,int);
	void create3DMesh(int,int,int);
	void create3DMesh(int);
	void init(array3<double> &);
	void getMemory();


	class crfft3d_mpi{
		pfft_plan plan_dft;
		fftw3mpi * myfftx{nullptr};
	public:
		crfft3d_mpi(fftw3mpi &);
		void fft(array3<Complex> & b,array3<double>  & a);
		void fftnormalize(array3<Complex> & b,array3<double>  & a);
	};

	class rcfft3d_mpi{
		pfft_plan plan_dft;
		fftw3mpi * myfftx{nullptr};
	public:
		rcfft3d_mpi(fftw3mpi &);
		void fft(array3<double>  & a, array3<Complex> & b);
	};



	MPI_Comm comm_cart_3d(){return *comm_cart_3dx;}


	ptrdiff_t * loc_ni(){return local_ni;}
	ptrdiff_t * loc_i_start(){return local_i_start;}
	ptrdiff_t * loc_no(){return local_no;}
	ptrdiff_t * loc_o_start(){return local_o_start;}
	virtual ~fftw3mpi();
};

} /* namespace Parallel */

#endif /* PARALLEL_FFTW3MPI_H_ */
