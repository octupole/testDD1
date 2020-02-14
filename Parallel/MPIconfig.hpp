/*
 * MPIconfig.hpp
 *
 *  Created on: May 29, 2019
 *      Author: marchi
 */

#ifndef PARALLEL_MPICONFIG_HPP_
#define PARALLEL_MPICONFIG_HPP_

namespace Parallel{

enum class Cartesian {up,down,left,right,north,south};

const int UP{static_cast<int>(Cartesian::up)};
const int DOWN{static_cast<int>(Cartesian::down)};
const int LEFT{static_cast<int>(Cartesian::left)};
const int RIGHT{static_cast<int>(Cartesian::right)};
const int NORTH{static_cast<int>(Cartesian::north)};
const int SOUTH{static_cast<int>(Cartesian::south)};

const int CARTDIRS{2};

const Cartesian UD[2]{Cartesian::up,Cartesian::down};
const Cartesian LR[2]{Cartesian::left,Cartesian::right};
const Cartesian NS[2]{Cartesian::north,Cartesian::south};
}

#endif /* PARALLEL_MPICONFIG_HPP_ */
