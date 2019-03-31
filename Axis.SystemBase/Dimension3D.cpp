#include "Dimension3D.hpp"

axis::Dimension3D::Dimension3D( void )
{
  X = 0; Y = 0; Z = 0;
}

axis::Dimension3D::Dimension3D(unsigned int x, unsigned int y, unsigned int z) : 
  X(x), Y(y), Z(z)
{
  // nothing to do here
}

axis::Dimension3D::Dimension3D( unsigned int x, unsigned int y ) : 
  X(x), Y(y), Z(0)
{
  // nothing to do here
}

axis::Dimension3D::Dimension3D( unsigned int x ) : 
  X(x), Y(0), Z(0)
{
  // nothing to do here
}

axis::Dimension3D::Dimension3D( const Dimension3D& other )
{
  X = other.X; Y = other.Y; Z = other.Z;
}

axis::Dimension3D& axis::Dimension3D::operator=( const Dimension3D& other )
{
  X = other.X; Y = other.Y; Z = other.Z;
  return *this;
}

axis::Dimension3D::~Dimension3D( void )
{
  // nothing to do here
}
