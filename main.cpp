#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include <random>
#include <fstream>
#include <sstream>

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

inline void init(array3<double> f, split3 d)
{
	  int rank;
	  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	  unsigned seed = 123456+rank;
	  std::default_random_engine generator (seed);
	  std::uniform_real_distribution<double> distribution (-100.0,100.0);

  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.y; j++) {
      unsigned int jj=d.y0+j;
      for(unsigned int k=0; k < d.Z; k++) {
          f(i,j,k)=ii+jj+k+1;
//          f(i,j,k)=distribution(generator);
      }
    }
  }
}
inline void init(array3<double> f)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank) return;
	unsigned seed = 123456;
	std::default_random_engine generator (seed);
	std::uniform_real_distribution<double> distribution (-100.0,100.0);

	for(unsigned int i=0; i < f.Nx(); ++i) {
		for(unsigned int j=0; j < f.Ny(); j++) {
			for(unsigned int k=0; k < f.Nz(); k++) {
//				f(i,j,k)=distribution(generator);
				f(i,j,k)=i+j+k+1;
			}
		}
	}
}
// Gather an MPI-distributed array onto the rank 0 process.
// The distributed array has dimensions x*y*Z.
// The gathered array has dimensions    X*Y*Z.
template<class ftype>
void scatterxy(const ftype *whole,
			ftype *part,
			const split3 d,
			const MPI_Comm& communicator)
{
  int size, rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  if(size ==1) return;
  const unsigned int Y=d.Y;
  const unsigned int Z=d.Z;
  const unsigned int x=d.x;
  const unsigned int y=d.yz.x;
  const unsigned int x0=d.x0;
  const unsigned int y0=d.yz.x0;

  if(rank == 0){
      unsigned int n=x*y*Z;
      for(unsigned int i=0; i < x; ++i) {
    	  const int inoffset=(x0+i)*Y*Z+y0*Z;
    	  const int outoffset=i*y*Z;
    	  for(unsigned int j=0; j < y; ++j)
    		  for(unsigned int k=0; k < Z; ++k){
    			  *(part+outoffset+j*Z+k)=*(whole+inoffset+j*Z+k);
    		  }
      }
      for(int p=1; p < size; ++p) {
          unsigned int dims[4];
          MPI_Recv(&dims,4,MPI_UNSIGNED,p,0,communicator,MPI_STATUS_IGNORE);
          unsigned int x=dims[0];
          unsigned int y=dims[1];
          unsigned int x0=dims[2];
          unsigned int y0=dims[3];
          unsigned int n=x*y*Z;
          if(n > 0) {
              ftype *C=new ftype[n];
              for(unsigned int i=0; i < x; ++i) {
            	  const int inoffset=(x0+i)*Y*Z+y0*Z;
            	  const int outoffset=i*y*Z;
            	  for(unsigned int j=0; j < y; ++j)
            		  for(unsigned int k=0; k < Z; ++k){
            			  *(C+outoffset+j*Z+k)=*(whole+inoffset+j*Z+k);
            		  }
              }
              MPI_Send(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator);
              delete []C;

          }

      }
  } else {
    unsigned int dims[] = {x,y,x0,y0};
    MPI_Send(&dims,4,MPI_UNSIGNED,0,0,communicator);
    unsigned int n=x*y*Z;
    if(n > 0)
      MPI_Recv((ftype *) part,n*sizeof(ftype),MPI_BYTE,0,0,communicator,MPI_STATUS_IGNORE);
  }

}

template<class ftype>
void mygatheryz(const ftype *part, ftype *whole, const split3& d,
              const MPI_Comm& communicator);

int main(int argc, char* argv[])
{
	unsigned int outlimit=3000;

#ifndef __SSE2__
	fftw::effort |= FFTW_NO_SIMD;
#endif
	int retval=0;

	// Default number of iterations.
	unsigned int N0=10000000;
	unsigned int N=0;
	int divisor=0; // Test for best block divisor
	int alltoall=-1; // Test for best alltoall routine

	unsigned int nx=4;
	unsigned int ny=0;
	unsigned int nz=0;

	bool inplace=false;

	bool quiet=false;
	bool test=false;
	bool shift=false;
	unsigned int stats=0; // Type of statistics used in timing test.

	int provided;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank != 0) opterr=0;
#ifdef __GNUC__
	optind=0;
#endif
	for (;;) {
		int c = getopt(argc,argv,"S:hti:N:O:T:a:i:m:n:s:x:y:z:q");
		if (c == -1) break;

		switch (c) {
		case 0:
			break;
		case 'a':
			divisor=atoi(optarg);
			break;
		case 'i':
			inplace=atoi(optarg);
			break;
		case 'N':
			N=atoi(optarg);
			break;
		case 'm':
			nx=ny=nz=atoi(optarg);
			break;
		case 's':
			alltoall=atoi(optarg);
			break;
		case 'x':
			nx=atoi(optarg);
			break;
		case 'y':
			ny=atoi(optarg);
			break;
		case 'z':
			nz=atoi(optarg);
			break;
		case 'n':
			N0=atoi(optarg);
			break;
		case 'O':
			shift=atoi(optarg);
			break;
		case 'S':
			stats=atoi(optarg);
			break;
		case 'T':
			fftw::maxthreads=atoi(optarg);
			break;
		case 'q':
			quiet=true;
			break;
		case 't':
			test=true;
			break;
		case 'h':
		default:
			if(rank == 0) {
				usage(3);
				usageTranspose();
				usageShift();
			}
			exit(1);
		}
	}

	if(ny == 0) ny=nx;
	if(nz == 0) nz=nx;

	if(N == 0) {
		N=N0/nx/ny/nz;
		if(N < 10) N=10;
	}

	unsigned int nzp=nz/2+1;
	MPIgroup group(MPI_COMM_WORLD,nx,ny,true);

	if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
		fftw::maxthreads=1;

	defaultmpithreads=fftw::maxthreads;

	if(group.rank < group.size) {
		bool main=group.rank == 0;
		bool main1=group.rank == 1;
		bool main2=group.rank == 2;
		if(!quiet && main) {
			cout << "Configuration: "
					<< group.size << " nodes X " << fftw::maxthreads
					<< " threads/node" << endl;
			cout << "Using MPI VERSION " << MPI_VERSION << endl;
			cout << "nx=" << nx << ", ny=" << ny << ", nz=" << nz << ", nzp=" << nzp
					<< endl;
		}

		bool showresult = nx*ny < outlimit;

		split3 df(nx,ny,nz,group);
		split3 dg(nx,ny,nzp,group,true);

		unsigned int dfZ=inplace ? 2*dg.Z : df.Z;

		split3 dfgather(nx,ny,dfZ,group);
		array3<Complex> g(dg.x,dg.y,dg.Z,ComplexAlign(dg.n));
		array3<double> f;
		if(inplace)
			f.Dimension(df.x,df.y,2*dg.Z,(double *) g());
		else
			f.Dimension(df.x,df.y,df.Z,doubleAlign(df.n));

		rcfft3dMPI rcfft(df,dg,f,g,mpiOptions(divisor,alltoall));

		if(!quiet && group.rank == 0)
			cout << "Initialized after " << seconds() << " seconds." << endl;

		size_t align=sizeof(Complex);
		if(test) {
			array3<double> flocal;
			flocal.Allocate(nx,ny,nz,align);
			//    	init(f,df);
			init(flocal);
			double t1=MPI_Wtime();
			scatterxy(flocal(),f(),df,group.active);
			double t2=MPI_Wtime();
			cout << t2-t1 <<endl;
			if(!quiet && showresult) {
				if(main)
					cout << "\nDistributed input:" << endl;
				show(f(),dfgather.x,dfgather.y,dfgather.Z,group.active);
			}

			array3<Complex> ggather(nx,ny,nzp,align);
			array3<Complex> glocal(nx,ny,nzp,align);
			array3<double> fgather(dfgather.X,dfgather.Y,dfgather.Z,align);
			rcfft3d localForward(nx,ny,nz,flocal,glocal);
			crfft3d localBackward(nx,ny,nz,glocal,flocal);

			gatherxy(f(), flocal(), dfgather, group.active);
			gatherxy(f(), fgather(), dfgather, group.active);

			if(main && !quiet)
				cout << "Gathered input:\n" <<  fgather << endl;

			rcfft.Forward(f,g);
			if(main) {
				localForward.fft(flocal,glocal);
				cout << endl;
			}

			if(!quiet && showresult) {
				if(main)
					cout << "Distributed output:" << endl;
				show(g(),dg.X,dg.y,dg.z,group.active);
			}

			gatheryz(g(),ggather(),dg,group.active);
			if(!quiet && main)
				cout << "Gathered output:\n" <<  ggather << endl;

			if(!quiet && main)
				cout << "Local output:\n" <<  glocal << endl;

			if(main)
				retval += checkerror(glocal(),ggather(),dg.X*dg.Y*dg.Z);

			rcfft.Backward(g,f);
			rcfft.Normalize(f);

			if(!quiet && showresult) {
				if(main)
					cout << "Distributed back to input:" << endl;
				show(f(),dfgather.x,dfgather.y,dfgather.Z,group.active);
			}

			gatherxy(f(),fgather(),dfgather,group.active);
			if(!quiet && main)
				cout << "Gathered back to input:\n" <<  fgather << endl;

			if(main) {
				localBackward.fftNormalized(glocal,flocal);
			}

			if(!quiet && main)
				cout << "Local back to input:\n" <<  flocal << endl;

			if(main)
				retval += checkerror(flocal(),fgather(),df.Z,df.X*df.Y,dfgather.Z);

			if(!quiet && group.rank == 0) {
				cout << endl;
				if(retval == 0)
					cout << "pass" << endl;
				else
					cout << "FAIL" << endl;
			}

		} else {
			if(main) cout << "N=" << N << endl;
			double *T=new double[N];

			for(unsigned int i=0; i < N; ++i) {
				array3<double> flocal;
				flocal.Allocate(nx,ny,nz,align);
				//    	init(f,df);
				double t1=MPI_Wtime();
				seconds();
				init(flocal);
				scatterxy(flocal(),f(),df,group.active);

//				init(f,df);
				if(shift) {
					rcfft.Forward0(f,g);
					rcfft.Backward0(g,f);
//					T[i]=0.5*seconds();
					rcfft.Normalize(f);
				} else {
					rcfft.Forward(f,g);
					rcfft.Backward(g,f);
//					T[i]=0.5*seconds();
					rcfft.Normalize(f);
				}
				double t2=MPI_Wtime();
				T[i]=t2-t1;
			}
			if(!quiet && showresult)
				show(f(),df.x,df.y,dfZ,0,0,0,df.x,df.y,df.Z,group.active);

			if(main) timings("FFT timing:",nx,T,N,stats);
			delete[] T;
		}

		deleteAlign(g());
		if(!inplace) deleteAlign(f());
	}

	MPI_Finalize();

	return retval;
}
// Gather an MPI-distributed array onto the rank 0 process.
// The distributed array has dimensions X*y*z.
// The gathered array has dimensions    X*Y*Z.
template<class ftype>
void mygatheryz(const ftype *part, ftype *whole, const split3& d,
              const MPI_Comm& communicator)
{
  int size, rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);

  const unsigned int X=d.X;
  const unsigned int Y=d.Y;
  const unsigned int Z=d.Z;
  const unsigned int y=d.xy.y;
  const unsigned int z=d.z;
  const unsigned int y0=d.xy.y0;
  const unsigned int z0=d.z0;
  cout << "Rank = "<<rank << " --> "<<d.X << " " <<y0<<" " << y<<" " << z0<< " " << z <<endl;
  if(rank == 1){
	  cout <<" P2 " <<  X << " " << Y << " " << Z <<endl;
      unsigned int n=X*y*z;
      if(n > 0) {
    	  const int count=y;
    	  const int stride=Z;
    	  const int length=z;
    	  for(unsigned int i=0; i < X; ++i) {
    		  const int outoffset=i*Y*Z+y0*Z+z0;
    		  const int inoffset=i*y*z;
    		  for(unsigned int j=0; j < y; ++j)
    			  for(unsigned int k=0; k < z; ++k){
    				  cout << i << " " << y0+j << " " << z0+k << " " <<*(part+inoffset+j*z+k) << " -- "<< outoffset+j*stride+k <<endl;
    			  }

    	  }
      }
  }
  exit(1);
}
