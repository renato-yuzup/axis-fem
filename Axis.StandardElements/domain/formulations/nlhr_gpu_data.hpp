#pragma once
#include "yuzu/common/gpu.hpp"

class NLHR_GPUFormulation
{
public:
  real BMatrix[24];
  real InitialJacobianInverse[9];
  real UpdatedJacobianInverse[9];
  real UpdatedJacobianDeterminant;  
};
