#pragma once
#include "foundation/basic_types.hpp"

namespace axis { namespace domain { namespace materials {
namespace bilinear_plasticity_commands {

volatile struct BiLinearPlasticityGPUData
{
  real YieldStress;
  real HardeningCoefficient;
};

} } } } // namespace axis::domain::materials::bilinear_plasticity_commands
