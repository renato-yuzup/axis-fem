#pragma once
#include "foundation/basic_types.hpp"

namespace axis { namespace domain { namespace materials { 
namespace neohookean_commands {

struct NeoHookeanGPUData
{
  real LambdaCoefficient;
  real MuCoefficient;
};

} } } } // namespace axis::domain::materials::neohookean_commands
