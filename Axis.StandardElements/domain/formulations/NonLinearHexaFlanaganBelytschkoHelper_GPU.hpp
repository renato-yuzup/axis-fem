#pragma once
#include "NonLinearHexaFlanaganBelytschkoHelper.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/Node.hpp"
#include "foundation/blas/blas.hpp"

#define XN(x) nodeCoord[3*MakeIdx(nodeIdx, x) + 0]
#define YN(x) nodeCoord[3*MakeIdx(nodeIdx, x) + 1]
#define ZN(x) nodeCoord[3*MakeIdx(nodeIdx, x) + 2]

namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

namespace {
	inline void BuildNodeCoordinateMatrixForGPU(real * nodeCoord, 
		const ade::ElementGeometry& geometry)
	{
		for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
		{
			const ade::Node& node = geometry[nodeIdx];
			nodeCoord[3*nodeIdx + 0] = node.CurrentX();
			nodeCoord[3*nodeIdx + 1] = node.CurrentY();
			nodeCoord[3*nodeIdx + 2] = node.CurrentZ();
		}
	}

  inline void CalculateBMatrixForGPU(real *B, 
		const real * nodeCoord)
	{
		for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
		{
			// For the sake of clarity, the expressions below use one-based index so 
			// that it looks more similar to Eq. (79) in Flanagan, Belytschko (1981).
			real B1X = ( YN(2)*( (ZN(6)-ZN(3)) - (ZN(4)-ZN(5)) )  +  YN(3)*(ZN(2)-ZN(4))
				+ YN(4)*( (ZN(3)-ZN(8)) - (ZN(5)-ZN(2)) )  
				+  YN(5)*( (ZN(8)-ZN(6)) - (ZN(2)-ZN(4)) )
				+ YN(6)*(ZN(5)-ZN(2)) + YN(8)*(ZN(4)-ZN(5)) 
				) / (real)12.0;
			real B2X = ( ZN(2)*( (XN(6)-XN(3)) - (XN(4)-XN(5)) )  +  ZN(3)*(XN(2)-XN(4))
				+ ZN(4)*( (XN(3)-XN(8)) - (XN(5)-XN(2)) )  
				+  ZN(5)*( (XN(8)-XN(6)) - (XN(2)-XN(4)) )
				+ ZN(6)*(XN(5)-XN(2)) + ZN(8)*(XN(4)-XN(5)) 
				) / (real)12.0;
			real B3X = ( XN(2)*( (YN(6)-YN(3)) - (YN(4)-YN(5)) )  +  XN(3)*(YN(2)-YN(4))
				+ XN(4)*( (YN(3)-YN(8)) - (YN(5)-YN(2)) )  
				+  XN(5)*( (YN(8)-YN(6)) - (YN(2)-YN(4)) )
				+ XN(6)*(YN(5)-YN(2)) + XN(8)*(YN(4)-YN(5)) 
				) / (real)12.0;
			B[nodeIdx] = B1X; B[8 + nodeIdx] = B2X; B[16 + nodeIdx] = B3X; 
		}
	}

	inline real CalculateElementVolumeForGPU(const real * B, 
		const real * nodeCoord)
	{
		real volume = 0;
		for (int i = 0; i < 8; i++)
		{
			volume += B[i] * nodeCoord[3*i];
		}
		return volume;
	}

} // namespace

#undef XN
#undef YN
#undef ZN
