/*
 * MyTools.cpp
 *
 *  Created on: May 15, 2019
 *      Author: marchi
 */

#include "MyTools.hpp"



string keywords::key="";
namespace forces{
double smoothstep::rout=9;
double smoothstep::rtol=1.0;
double smoothstep::rcut=10.0;
double erfc::alpha=0.3;
double erfc::rkcut=1.0;

}
namespace Memory{
template <>
CurrentProcess<memoryEnum::Memory>::Memory CurrentProcess<memoryEnum::Memory>::get=&MemoryHelper::getValue;
template <>
CurrentProcess<memoryEnum::Vmemory>::Memory CurrentProcess<memoryEnum::Vmemory>::get=&MemoryHelper::getValue0;
}
