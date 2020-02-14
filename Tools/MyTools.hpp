/*
 * MyTools.hpp
 *
 *  Created on: May 15, 2019
 *      Author: marchi
 */

#ifndef TOOLS_MYTOOLS_HPP_
#define TOOLS_MYTOOLS_HPP_
#include <sstream>
#include <regex>
#include <set>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include "MyUtilClass.h"
#include <chrono>
#include <functional>
#include "../abseil-cpp/absl/strings/str_split.h"
#include "../abseil-cpp/absl/strings/string_view.h"

using uint=unsigned int;
using std::vector;
using std::string;
using std::cout;
using std::endl;
#include "MyHash.hpp"
#include "Dimensions.hpp"
using std::istringstream;
using std::istream_iterator;

#include <xmmintrin.h>
constexpr double M_SQRTPI{2/M_2_SQRTPI};
static auto ssplit=[](string text) -> std::vector<std::string> {
	return absl::StrSplit(text,absl::ByAnyChar(" \t"),absl::SkipEmpty());
};

namespace alloc{
template<class T, uint N>
struct helperDims
{
	static void Deallocate(T & x,uint PP){
		for(size_t o{0};o<PP;o++){
			helperDims<decltype(x[o]),N-1>::Deallocate(x[o],PP);
		}
		delete [] x;
		x=nullptr;
	}

	static void Allocate(T & x, uint PP){
		auto spaced=sizeof(T)*PP;
		void * pt{nullptr};
		pt=_mm_malloc(spaced,32);
		x=reinterpret_cast<T>(pt);
		for(size_t o{0};o<PP;o++){
			helperDims<decltype(x[o]),N-1>::Allocate(x[o],PP);
		}
	}
};
template<class T>
struct helperDims<T,0>
{
	static void Deallocate(T & x, uint PP){

	}
	static void Allocate(T & x, uint PP){
		return;
	}
};
template <uint N,typename T>
static void Allocate(T & x, uint PP){
	helperDims<T,N>::Allocate(x,PP);
}
template <uint N,typename T>
static void Deallocate(T & x, uint PP){
	helperDims<T,N>::Deallocate(x,PP);
}


}
struct keywords{
	keywords(const vector<string> s,string pre=R"(^)",string post=R"(.*$)"){
		preKey=pre;postKey=post;
		for(auto it: s){
			keys.insert(pre+it+post);
		}
	}
	string operator()(string str){
		auto npre=preKey.size();
		auto npost=postKey.size();
		for(auto it=keys.begin();it != keys.end();it++){
			std::regex pat{*it};
			std::smatch matches;
			if(std::regex_search(str,matches,pat)){
				key=it->substr(npre,it->size()-npost-1);
			}
		}
		return key;
	}
static string key;
private:
	string preKey,postKey;
	std::set<string> keys;
};




template<class T, uint N>
struct helper
{
	static constexpr T pow(const T x){
		return helper<T, N-1>::pow(x) * x;
	}
};

template<class T>
struct helper<T, 0>
{
	static constexpr T pow(const T x){
		return 1;
	}
};
template<uint N, class T>
T constexpr pow(T const x)
{
	return helper<T, N>::pow(x);
}
constexpr double epsilon{0.02},a2{1.0/6},a4{1.0/120.0},a6{1.0/5040.0};
template <typename T>
constexpr T SinhOverX(T x){
	return x>epsilon?sinh(x)/x:1.0+pow<2>(x)*a2+pow<4>(x)*a4+pow<6>(x)*a6;
}

constexpr double swrs(double & rout,double & rtol,double x){
	return x>rtol+rout?0:x<rout?1:1-pow<2>((x-rout)/rtol)*(3-2*(x-rout)/rtol);
}
constexpr double dswrs(double & rout, double & rtol,double x){
	return x>rtol+rout||x<rout?0:-(2*(x-rout)/rtol)*(3-2*(x-rout)/rtol)/rtol+2.0*pow<2>((x-rout)/rtol)/rtol;
}
namespace Memory{
struct MemoryHelper{
	static int parseLine(char* line){
	    // This assumes that a digit will be found and the line ends in " Kb".
	    int i = strlen(line);
	    const char* p = line;
	    while (*p <'0' || *p > '9') p++;
	    line[i-3] = '\0';
	    i = atoi(p);
	    return i;
	}
	// Virtual memory from the process
	static double getValue0(){ //Note: this value is in KB!
	    FILE* file = fopen("/proc/self/status", "r");
	    int result = -1;
	    char line[128];

	    while (fgets(line, 128, file) != NULL){
	        if (strncmp(line, "VmSize:", 7) == 0){
	            result = parseLine(line);
	            break;
	        }
	    }
	    fclose(file);
	    return (double) result/ 1024.0/1024.0;
	}
	// Physical memory from the process
	static double getValue(){ //Note: this value is in KB!
	    FILE* file = fopen("/proc/self/status", "r");
	    int result = -1;
	    char line[128];

	    while (fgets(line, 128, file) != NULL){
	        if (strncmp(line, "VmRSS:", 6) == 0){
	            result = parseLine(line);
	            break;
	        }
	    }
	    fclose(file);
	    return (double) result/1024.0/1024.0;
	}
};
enum memoryEnum{Vmemory, Memory};

template <memoryEnum type>
struct CurrentProcess{
	typedef double (*Memory)();
	static Memory get;
};

}
namespace outputs{
enum infosimEnum {inter,other};
template <infosimEnum type>
struct InfoSim{
	InfoSim(){
		switch(type){
		case infosimEnum::inter:
			std::cout << std::setprecision(5)<<std::setw(14)<<std::right;
			break;
		case infosimEnum::other:
			std::cout << std::setprecision(5)<<std::setw(14)<<std::right ;
			break;
		default:
			std::cout << "";
		}
	}
	template <typename T>
	const InfoSim & operator<<(const T & v) const {
		std::cout << v;
		return *this;
	}
};
struct Log
{
    Log(string str="Info")
    {
        std::cout << str+": ";
    }

    template <class T>
    const Log &operator<<(const T &v) const
    {
        std::cout << v;
        return *this;
    }

    ~Log()
    {
        std::cout << std::endl;
    }
};

}
namespace forces{
using Dvect=DVECT::DDvect<double>;
struct zero{
	static void forces(vector<Dvect> & f){
		for(size_t o{0};o< f.size();o++)f[o]=Dvect{0};
	}
};
constexpr size_t LJindx(size_t n, size_t m){
	return n>m?n*(n+1)/2+m:m*(m+1)/2+n;
}


struct smoothstep{
	static double rout;
	static double rtol;
	static double rcut;

	static constexpr double sw(double x){
		return swrs(rout,rtol,x);
	}
	static constexpr double dsw(double x){
		return dswrs(rout,rtol,x);
	}
};

constexpr double qp{0.3275911},a1{0.2548296},a2{-0.28449674},a3{1.4214137},a4{-1.453152},a5{1.0614054},twrtpi{2/M_SQRTPI};
constexpr double alpha_e(double & alpha){
	return alpha;
};
constexpr double alphar_e(double & alpha,double x){
	return alpha*x;
};

constexpr double qt_e(double & alpha,double x){
	return 1/(1+qp*alpha*x);
};
constexpr double expcst_e(double & alpha, double x){
	return exp(-pow<2>(alpha*x));
}

struct erfc{
	static double alpha;
	static double rkcut;
	static constexpr double alphar(double x){
		return alphar_e(alpha,x);
	};
	static constexpr double qt(double x){
		return qt_e(alpha,x);
	};
	static constexpr double expcst(double x){
		return expcst_e(alpha,x);
	}
	static constexpr double erfcst(double x){
		return ((((a5*qt(x)+a4)*qt(x)+a3)*qt(x)+a2)*qt(x)+a1)*qt(x)*expcst(x);
	}
	static constexpr double erfst(double x){
		return 1-erfcst(x);
	}
	static constexpr double erfc_ri(double x){
		return ((((a5*qt(x)+a4)*qt(x)+a3)*qt(x)+a2)*qt(x)+a1)*qt(x)*expcst(x)/x;
	}

	static constexpr double derfcst(double x){
		return -twrtpi*alphar(1)*expcst(x);
	}
	static constexpr double derfst(double x){
		return twrtpi*alphar(1)*expcst(x);
	}
	static constexpr double derfst2(double x){
		return -derfcst2(x);
	}
	static constexpr double derfcst2(double x){
		return -2*alphar(x)*alphar(1)*erfcst(x) - expcst(x)*((((5*a5*qt(x)+4*a4)*qt(x)
				+3*a3)*qt(x)+2*a2)*qt(x)+a1)*(qp*alphar(1)*pow<2>(qt(x)));
	}
	static constexpr double derfc_ri(double x){
		return (x*derfcst(x)-erfcst(x))/pow<2>(x);
	}
	static constexpr double derfc_ri2(double x){
		return (x*derfcst2(x)-erfcst(x))/pow<2>(x);
	}

};
template <typename T>
constexpr T GetVolume(const MATRIX::MMatrix<T> & co){
	return static_cast<T>(co[XX][XX]*(co[YY][YY]*co[ZZ][ZZ]-co[ZZ][YY]*co[YY][ZZ])
			  -co[YY][XX]*(co[XX][YY]*co[ZZ][ZZ]-co[ZZ][YY]*co[XX][ZZ])
			  +co[ZZ][XX]*(co[XX][YY]*co[YY][ZZ]-co[YY][YY]*co[XX][ZZ]));
}
template<typename T, uint N> struct DcosN{
	static constexpr T dcn(const T x);
	static constexpr T cn(const T x);
};

template<typename T> struct DcosN<T,1>{
	static constexpr T  cn(const T x){return x;}
	static constexpr T dcn(const T x){return 1.0;}
};

template<typename T> struct DcosN<T,2>{
	static constexpr T  cn(const T x){return 2*pow<2>(x)-1;}
	static constexpr T dcn(const T x){return 4*x;}
};

template<typename T> struct DcosN<T,3>{
	static constexpr T  cn(const T x){return 4*pow<3>(x)-3.0*x;}
	static constexpr T dcn(const T x){return 12*pow<2>(x)-3.0;}
};

template<typename T> struct DcosN<T,4>{
	static constexpr T  cn(const T x){return 8*pow<4>(x)-8.0*pow<2>(x)+1;}
	static constexpr T dcn(const T x){return 32*pow<3>(x)-16.0*x;}
};

template<typename T> struct DcosN<T,5>{
	static constexpr T  cn(const T x){return 16*pow<5>(x)-20*pow<3>(x)+5*x;}
	static constexpr T dcn(const T x){return 16*5*pow<4>(x)-60*pow<2>(x)+5;}
};

template<typename T> struct DcosN<T,6>{
	static constexpr T  cn(const T x){return 32*pow<6>(x)-48*pow<4>(x)+18*pow<2>(x)-1;}
	static constexpr T dcn(const T x){return 32*6*pow<5>(x)-48*4*pow<3>(x)+36*x;}
};

template <uint N, typename T>
constexpr T dcosn(const T x){
	return DcosN<T,N>::dcn(x);
}
template <uint N, typename T>
constexpr T cosn(const T x){
	return DcosN<T,N>::cn(x);
}

inline Dvect cross(const double &k, const Dvect &v1, const Dvect &v2) {
   return Dvect( k*(v1[YY]*v2[ZZ]-v2[YY]*v1[ZZ]),
                  k*(v2[XX]*v1[ZZ]-v1[XX]*v2[ZZ]),
                  k*(v1[XX]*v2[YY]-v2[XX]*v1[YY]) );
 }
}
namespace SimBox{
constexpr double RAD2DEG=180.0/M_PI;

template <class T>
constexpr static T norm2(const T a[DIM]){
  return a[XX]*a[XX]+a[YY]*a[YY]+a[ZZ]*a[ZZ];}

template <class T>
constexpr static T norm(const T a[DIM]){
  return static_cast<T>(sqrt(a[XX]*a[XX]+a[YY]*a[YY]+a[ZZ]*a[ZZ]));}

template <typename T>
struct cellPar{
	static inline T cos_angle(const T a[DIM],const T b[DIM]){
	  T   cosval;
	  int    m;
	  double aa,bb,ip,ipa,ipb;
	  ip=ipa=ipb=0.0;
	  for(m=0; (m<DIM); m++) {
	    aa   = a[m];
	    bb   = b[m];
	    ip  += aa*bb;
	    ipa += aa*aa;
	    ipb += bb*bb;
	  }
	  cosval=ip/sqrt(ipa*ipb);
	  if (cosval > 1.0)
	    return  1.0;
	  if (cosval <-1.0)
	    return -1.0;

	  return cosval;
	}
	T a{0},b{0},c{0},alpha{0},beta{0},gamma{0};
	cellPar(MATRIX::MMatrix<T> & CO){
		a=norm(CO[XX]);
		b=norm(CO[YY]);
		c=norm(CO[ZZ]);
		alpha=norm2(CO[YY])*norm2(CO[ZZ])!=0?acos(cos_angle(CO[YY],CO[ZZ])):0.5*M_PI;
		beta =norm2(CO[XX])*norm2(CO[ZZ])!=0?acos(cos_angle(CO[XX],CO[ZZ])):0.5*M_PI;
		gamma=norm2(CO[XX])*norm2(CO[YY])!=0?acos(cos_angle(CO[XX],CO[YY])):0.5*M_PI;
		alpha*=RAD2DEG;
		beta*=RAD2DEG;
		gamma*=RAD2DEG;
	}
};


}

#endif /* TOOLS_MYTOOLS_HPP_ */
