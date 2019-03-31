#pragma once
#include "yuzu/common/gpu.hpp"

class LHR_GPUFormulation
{
public:
  real BMatrix[144];
  real JacobianDeterminant;
  bool InitializedBMatrices;
};
