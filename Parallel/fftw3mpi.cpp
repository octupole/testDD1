/*
 * fftw3mpi.cpp
 *
 *  Created on: Feb 11, 2020
 *      Author: marchi
 */

#include "fftw3mpi.h"

namespace Parallel {


fftw3mpi::fftw3mpi(int nx, int ny, int nz) {
	int isInitialized;
	MPI_Initialized(&isInitialized);
	try{
	if(!isInitialized) throw string("MPI must be intialized before calling pfft!");
	}catch(const string & s){
		cout << s << endl;
		MPI_Finalize();
	}
	pfft_init();
	NN[XX]=nx;NN[YY]=ny;NN[ZZ]=nz;
}
fftw3mpi::fftw3mpi() {
	int isInitialized;
	MPI_Initialized(&isInitialized);
	try{
	if(!isInitialized) throw string("MPI must be intialized before calling pfft!");
	}catch(const string & s){
		cout << s << endl;
		MPI_Finalize();
	}
	pfft_init();
}
bool fftw3mpi::isInitialized(){
	bool ok=true;
	for(size_t o{0};o<DIM;o++){
		ok&=np[o]>0;
		ok&=NN[o]>0;
	}
	return ok;
}
void fftw3mpi::setDimFFT(int nx,int ny, int nz){
	NN[XX]=nx;NN[YY]=ny;NN[ZZ]=nz;
}
void fftw3mpi::getMemory(){
	try{
		if(!isInitialized()) throw string("fftw3mpi not initialized");
	}catch(const string & s){cout << s <<endl;MPI_Finalize();exit(0);}

	auto alloc_local=pfft_local_size_dft_r2c_3d(NN, *comm_cart_3dx, PFFT_TRANSPOSED_NONE,
	      local_ni, local_i_start, local_no, local_o_start);
	in.Allocate(2*alloc_local);
	out.Allocate(alloc_local);
}
void fftw3mpi::create3DMesh(int nx, int ny, int nz){
	np[XX]=nx;
	np[YY]=ny;
	np[ZZ]=nz;
	int npp[DIM]={nx,ny,nz};
	comm_cart_3dx=new MPI_Comm;

	try{
	 if(pfft_create_procmesh(DIM, MPI_COMM_WORLD,npp, comm_cart_3dx)) throw string("Error: This test file only works with ");
	}catch(const string & s){
		cout << s << np[XX]*np[YY]*np[ZZ] << " processes "<<endl;
		MPI_Finalize();exit(0);
	}
}
void fftw3mpi::create3DMesh(int n){
	vector<int> np0=findSize3D(n);
	np[XX]=np0[XX];
	np[YY]=np0[YY];
	np[ZZ]=np0[ZZ];

	int npp[DIM]={np0[XX],np0[YY],np0[ZZ]};
	comm_cart_3dx=new MPI_Comm;
	try{
	 if(pfft_create_procmesh(DIM,  MPI_COMM_WORLD, npp, comm_cart_3dx)) throw string("Error: This test file only works with ");
	}catch(const string & s){
		cout << s << np[XX]*np[YY]*np[ZZ] << " processes "<<endl;
		MPI_Finalize();exit(0);
	}
}
void fftw3mpi::_copy_from_in(array3<double> & x){
	long int * lis = this->local_i_start;
	long int  * lni =this->local_ni;
	int m{0};

	for(int k0=lis[0]; k0<lis[0]+lni[0]; k0++)
		for(int k1=lis[1]; k1<lis[1]+lni[1]; k1++)
			for(int k2=lis[2]; k2<lis[2]+lni[2]; k2++, m++){
				x[k0][k1][k2]=in[m];

			}
}
void fftw3mpi::_copy_to_in(array3<double> & x){
	long int * lis = this->local_i_start;
	long int  * lni =this->local_ni;
	int m{0};

	for(int k0=lis[0]; k0<lis[0]+lni[0]; k0++)
		for(int k1=lis[1]; k1<lis[1]+lni[1]; k1++)
			for(int k2=lis[2]; k2<lis[2]+lni[2]; k2++, m++){
				in[m]=x[k0][k1][k2];
			}
}
void fftw3mpi::_copy_from_out(array3<Complex> & x){
	long int * los = this->local_o_start;
	long int  * lno =this->local_no;
	int m{0};

	for(int k0=los[0]; k0<los[0]+lno[0]; k0++)
		for(int k1=los[1]; k1<los[1]+lno[1]; k1++)
			for(int k2=los[2]; k2<los[2]+lno[2]; k2++, m++){
				x[k0][k1][k2]=out[m];
			}
}
void fftw3mpi::_copy_to_out(array3<Complex> & x){
	long int * los = this->local_o_start;
	long int  * lno =this->local_no;
	int m{0};

	for(int k0=los[0]; k0<los[0]+lno[0]; k0++)
		for(int k1=los[1]; k1<los[1]+lno[1]; k1++)
			for(int k2=los[2]; k2<los[2]+lno[2]; k2++, m++){
				out[m]=x[k0][k1][k2];
			}
}
void fftw3mpi::init(array3<double> & x){
	long int * lis = this->local_i_start;
	long int  * lni =this->local_ni;

	pfft_init_input_real(3, NN, lni, lis,&in[0]);
	_copy_from_in(x);

}
fftw3mpi::rcfft3d_mpi::rcfft3d_mpi(fftw3mpi & x){
	try{
		if(!x.isInitialized()) throw string("fftw3mpi not initialized");
	}catch(const string & s){cout << s <<endl;MPI_Finalize();exit(0);}
	myfftx=&x;
	double * in=&x.in[0];
	fftw_complex * out=(fftw_complex *) &x.out[0];

	rcfft3d_mpi::plan_dft=pfft_plan_dft_r2c_3d(x.NN,in, out, x.comm_cart_3d(),
			PFFT_FORWARD, PFFT_TUNE|PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_PRESERVE_INPUT);
}
fftw3mpi::crfft3d_mpi::crfft3d_mpi(fftw3mpi & x){
	myfftx=&x;
	double * in=&x.in[0];
	fftw_complex * out=(fftw_complex *) &x.out[0];

	crfft3d_mpi::plan_dft=pfft_plan_dft_c2r_3d(x.NN, out, in, x.comm_cart_3d(),
			PFFT_BACKWARD, PFFT_TUNE|PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_PRESERVE_INPUT);
}
void fftw3mpi::rcfft3d_mpi::fft(array3<double> & A, array3<Complex> & B){
	myfftx->_copy_to_in(A);
	myfftx->out=Complex{0.0,0.0};
	auto in=&myfftx->in[0];
	auto out=&myfftx->out[0];
	pfft_execute(rcfft3d_mpi::plan_dft);
	myfftx->_copy_from_out(B);
}

void fftw3mpi::crfft3d_mpi::fft(array3<Complex> & B,array3<double> & A){
	myfftx->_copy_to_out(B);
	myfftx->in=0.0;
	pfft_execute(crfft3d_mpi::plan_dft);
	myfftx->_copy_from_in(A);
}
void fftw3mpi::crfft3d_mpi::fftnormalize(array3<Complex> & B,array3<double> & A){
	long int  * lni =myfftx->local_ni;
	myfftx->_copy_to_out(B);
	myfftx->in=0.0;
	pfft_execute_dft_c2r(crfft3d_mpi::plan_dft,(fftw_complex *)&myfftx->out[0],&myfftx->in[0]);
	auto NN=myfftx->NN;
	for(int l=0; l < lni[0] * lni[1] * lni[2]; l++){
	    myfftx->in[l] /= (NN[XX]*NN[YY]*NN[ZZ]);
	}
	myfftx->_copy_from_in(A);

}

fftw3mpi::~fftw3mpi() {}

} /* namespace Parallel */
