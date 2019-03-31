#pragma once
#include "stdafx.h"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/Node.hpp"
#include "foundation/blas/blas.hpp"

#define XN(x) nodeCoord(MakeIdx(nodeIdx, x), 0)
#define YN(x) nodeCoord(MakeIdx(nodeIdx, x), 1)
#define ZN(x) nodeCoord(MakeIdx(nodeIdx, x), 2)

namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

namespace {
	const real hourglassVectors[4][8] = {{ 1,  1, -1, -1, -1, -1, 1,  1},
	{ 1, -1, -1,  1, -1,  1, 1, -1},
	{ 1, -1,  1, -1,  1, -1, 1, -1},
	{-1,  1, -1,  1,  1, -1, 1, -1}};

	inline void BuildNodeCoordinateMatrix(afb::DenseMatrix& nodeCoord, 
		const ade::ElementGeometry& geometry)
	{
		for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
		{
			const ade::Node& node = geometry[nodeIdx];
			nodeCoord(nodeIdx, 0) = node.CurrentX();
			nodeCoord(nodeIdx, 1) = node.CurrentY();
			nodeCoord(nodeIdx, 2) = node.CurrentZ();
		}
	}

	inline int MakeIdx(int baseIdx, int offset)
	{
		if (baseIdx < 4)
			return (offset < 5) ? (baseIdx + offset-1) % 4 :
			4 + ((baseIdx + offset-5)) % 4;
		else
			return (offset < 5) ? 4 + (baseIdx - offset + 5) % 4 :
			(baseIdx - offset + 5) % 4;
	}

	inline void CalculateBMatrix(afb::DenseMatrix& B, 
		const afb::DenseMatrix& nodeCoord)
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
			B(0, nodeIdx) = B1X; B(1, nodeIdx) = B2X; B(2, nodeIdx) = B3X; 
		}
	}

	inline void CalculateBMatrixForGPU(real *B, 
		const afb::DenseMatrix& nodeCoord)
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

	inline real CalculateElementVolume(const afb::DenseMatrix& B, 
		const afb::DenseMatrix& nodeCoord)
	{
		real volume = 0;
		for (int i = 0; i < 8; i++)
		{
			volume += B(0,i) * nodeCoord(i, 0);
		}
		return volume;
	}

} // namespace

#undef XN
#undef YN
#undef ZN
