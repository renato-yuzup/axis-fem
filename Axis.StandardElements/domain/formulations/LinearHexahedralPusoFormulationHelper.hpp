#pragma once
#include "domain/elements/ElementGeometry.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "foundation/axis.CommonLibrary.hpp"
#include "Foundation/BLAS/DenseMatrix.hpp"

namespace adf = axis::domain::formulations;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace af = axis::foundation;
namespace afb = axis::foundation::blas;

#define XN(x) nodeCoord(MakeIdx(nodeIdx, x), 0)
#define YN(x) nodeCoord(MakeIdx(nodeIdx, x), 1)
#define ZN(x) nodeCoord(MakeIdx(nodeIdx, x), 2)

namespace {

const real csiVectors_[3][8] = {{-1,  1,  1, -1, -1,  1, 1, -1}, 
                                {-1, -1,  1,  1, -1, -1, 1,  1}, 
                                {-1, -1, -1, -1,  1,  1, 1,  1}};

const real hourglassVector_[4][8] = {{ 1,  1, -1, -1, -1, -1, 1,  1},
                                     { 1, -1, -1,  1, -1,  1, 1, -1},
                                     { 1, -1,  1, -1,  1, -1, 1, -1},
                                     {-1,  1, -1,  1,  1, -1, 1, -1}};

/*
  * This function permutes node numbering, just like Table 3 in 
  * Flanagan-Belytschko (1981) paper.
**/
inline int MakeIdx(int baseIdx, int offset)
{
  if (baseIdx < 4)
    return (offset < 5) ? (baseIdx + offset-1) % 4 :
    4 + ((baseIdx + offset-5)) % 4;
  else
    return (offset < 5) ? 4 + (baseIdx - offset + 5) % 4 :
    (baseIdx - offset + 5) % 4;
}

/*
  * Given node coordinates, calculates a 3x8 B matrix.
**/
inline void CalculateBMatrix(afb::DenseMatrix& B, real& volume, 
  const afb::DenseMatrix& nodeCoord)
{
  // Calculate B' = Ve*B
  for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
  {
    // For the sake of clarity, the expressions below use one-based index so 
    // that it looks more similar to Eq. (79) in Flanagan, Belytschko (1981).
    real B1X = ( YN(2)*( (ZN(6)-ZN(3)) - (ZN(4)-ZN(5)) )  +  YN(3)*(ZN(2)-ZN(4))
      + YN(4)*( (ZN(3)-ZN(8)) - (ZN(5)-ZN(2)) )  
      + YN(5)*( (ZN(8)-ZN(6)) - (ZN(2)-ZN(4)) )
      + YN(6)*(ZN(5)-ZN(2)) + YN(8)*(ZN(4)-ZN(5)) 
      ) / (real)12.0;
    real B2X = ( ZN(2)*( (XN(6)-XN(3)) - (XN(4)-XN(5)) )  +  ZN(3)*(XN(2)-XN(4))
      + ZN(4)*( (XN(3)-XN(8)) - (XN(5)-XN(2)) )  
      + ZN(5)*( (XN(8)-XN(6)) - (XN(2)-XN(4)) )
      + ZN(6)*(XN(5)-XN(2)) + ZN(8)*(XN(4)-XN(5)) 
      ) / (real)12.0;
    real B3X = ( XN(2)*( (YN(6)-YN(3)) - (YN(4)-YN(5)) )  +  XN(3)*(YN(2)-YN(4))
      + XN(4)*( (YN(3)-YN(8)) - (YN(5)-YN(2)) )  
      + XN(5)*( (YN(8)-YN(6)) - (YN(2)-YN(4)) )
      + XN(6)*(YN(5)-YN(2)) + XN(8)*(YN(4)-YN(5)) 
      ) / (real)12.0;
    B(0, nodeIdx) = B1X; B(1, nodeIdx) = B2X; B(2, nodeIdx) = B3X; 
  }

  // Calculate volume using B'
  volume = 0;
  for (int i = 0; i < 8; i++)
  {
    volume += B(0,i) * nodeCoord(i, 0);
  }

  // Now calculate mean B = 1/Ve*B'
  B.Scale(1.0 / volume);
}

inline void CalculateNodeCoordinate(afb::DenseMatrix& nodeCoord, 
  const ade::ElementGeometry& geometry)
{
  for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
  {
    const ade::Node& node = geometry[nodeIdx];
    nodeCoord(nodeIdx, 0) = node.X();
    nodeCoord(nodeIdx, 1) = node.Y();
    nodeCoord(nodeIdx, 2) = node.Z();
  }
}

inline void CalculateJacobian(afb::DenseMatrix& jacobian, 
  const afb::DenseMatrix& nodeCoordinate)
{
  for (int rowIdx = 0; rowIdx < 3; rowIdx++)
  {
    for (int colIdx = 0; colIdx < 3; colIdx++)
    {
      real j_rc = 0;
      for (int i = 0; i < 8; i++)
      {
        real x = nodeCoordinate(i, rowIdx);
        real csi = csiVectors_[colIdx][i];
        j_rc += x*csi;
      }
      jacobian(rowIdx, colIdx) = 0.125 * j_rc;
    }
  }
}

} // namespace
