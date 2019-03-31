#pragma once
#include "yuzu/common/gpu.hpp"

class NLHRFB_GPUFormulation
{
public:
	real BMatrix[24];
	real InitialJacobianInverse[9];
	real NodePosition[24];
	real HourglassForces[24];
	real Volume;
	real AntiHourglassRatio;
};
