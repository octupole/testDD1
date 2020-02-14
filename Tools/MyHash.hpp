/*
 * MyHash.h
 *
 *  Created on: May 15, 2019
 *      Author: marchi
 */

#ifndef FORCEFIELD_MYHASH_HPP_
#define FORCEFIELD_MYHASH_HPP_
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <array>


using std::vector;
using std::string;
using std::cout;
using std::endl;
struct MyHash
{
	template <size_t N>
	size_t operator()(std::array<string,N> tokens ) const {
    	try{
    	switch(N){
    	case 1:
    		return (*this)(tokens[0]);
    		break;
    	case 2:
    		return (*this)(tokens[0],tokens[1]);
    		break;
    	case 3:
    		return (*this)(tokens[0],tokens[1],tokens[2]);
    		break;
    	case 4:
    		return (*this)(tokens[0],tokens[1],tokens[2],tokens[3]);
    		break;
    	default:
    		throw "Can have only 1 to 4 element in array in input.";
    	}
    	} catch(const string & s ){
    		cout << s << endl;
    		exit(1);
    	}

	}
    std::size_t operator()(vector<string> tokens) const
    {
    	try{
    	switch(tokens.size()){
    	case 1:
    		return (*this)(tokens[0]);
    		break;
    	case 2:
    		return (*this)(tokens[0],tokens[1]);
    		break;
    	case 3:
    		return (*this)(tokens[0],tokens[1],tokens[2]);
    		break;
    	case 4:
    		return (*this)(tokens[0],tokens[1],tokens[2],tokens[3]);
    		break;
    	default:
    		throw "Can have only 1 to 4 element vector in input.";
    	}
    	} catch(const string & s ){
    		cout << s << endl;
    		exit(1);
    	}
    }
	std::size_t operator()(string one) const noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(one);
        return h1;
    }
	std::size_t operator()(string one, string two) const noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(one);
        std::size_t h2 = std::hash<std::string>{}(two);
        return h1 == h2?h1:h1 ^ h2  ;
    }
    std::size_t operator()(string one, string two, string three) const noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(one);
        std::size_t h2 = std::hash<std::string>{}(two);
        std::size_t h3 = std::hash<std::string>{}(three);
        if(h1 == h2 && h2 == h3) return h1;
        if(h1 == h3) return h1^(h2<<1);
        return (h3 ^ h1)^ (h2 << 1); // symmetry around h2
    }
    std::size_t operator()(string one, string two, string three, string four) const noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(one);
        std::size_t h2 = std::hash<std::string>{}(two);
        std::size_t h3 = std::hash<std::string>{}(three);
        std::size_t h4 = std::hash<std::string>{}(four);
        if(h1 == h2 && h2 == h3 && h3 == h4) return h1;
        if(h1 == h4 && h2 == h3) return h1 ^ (( h2 << 1) );
        if(h1 == h4) return h1 ^ (( h2 << 1) )^ ((h3 << 1));
        if(h2 == h3) return h1 ^ ( h2 << 1)  ^  h4 ;
//        auto ha=h1^ (h2 << 1)^(h3<<2)^(h4<<3);
//        auto hb=ha^ (h4^ (h3 << 1)^(h2<<2)^(h1<<3));
//        return  hb;
       return  (h1 ^ (h2 <<1))&(h4 ^ (h3 << 1)); // dihedral and improper symmetry
    }
};



#endif /* FORCEFIELD_MYHASH_HPP_ */
