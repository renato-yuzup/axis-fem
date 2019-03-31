#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis {

class AXISSYSTEMBASE_API Dimension3D
{
public:
  Dimension3D(void);
  Dimension3D(unsigned int x, unsigned int y, unsigned int z);
  Dimension3D(unsigned int x, unsigned int y);
  Dimension3D(unsigned int x);
  Dimension3D(const Dimension3D& other);
  Dimension3D& operator =(const Dimension3D& other);
  ~Dimension3D(void);

  unsigned int X;
  unsigned int Y;
  unsigned int Z;
};

} // namespace axis
