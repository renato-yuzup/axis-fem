#include "MultiLineCurve.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "System.hpp"
#include "foundation/memory/pointer.hpp"
#include "MultiLineCurveCommand.hpp"

namespace adcu = axis::domain::curves;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

#define X(x)  points[2*(x) + 0]
#define Y(x)  points[2*(x) + 1]

static adcu::MultiLineCurveCommand multiLineGPUCommand;

adcu::MultiLineCurve::MultiLineCurve( size_t numPoints )
{
	if (numPoints == 0) throw axis::foundation::ArgumentException();
	numPoints_ = numPoints;
  pointArrayPtr_ = System::ModelMemory().Allocate(sizeof(uint64) + 
    2*numPoints * sizeof(real));
  uint64 *numPointsGPU_ = absptr<uint64>(pointArrayPtr_);
  *numPointsGPU_ = numPoints;
  points_ = (real *)(absptr<char>(pointArrayPtr_) + sizeof(uint64));
}

adcu::MultiLineCurve::~MultiLineCurve( void )
{
  // nothing to do here
}

void adcu::MultiLineCurve::Destroy( void ) const
{
	// nothing to do here
}

real adcu::MultiLineCurve::GetValueAt( real xCoord ) const
{
	return operator [](xCoord);
}

real adcu::MultiLineCurve::operator[]( real xCoord ) const
{
  const real *points = points_;
	for (size_t i = 0; i < numPoints_; i++)
	{
		if ((X(i) > xCoord) || (i == numPoints_-1 && (MATH_ABS(X(i) - xCoord) <= 1e-15)))
		{	
			if (i == 0) throw axis::foundation::ArgumentException();
			
			// trivial case: horizontal line
			if (MATH_ABS(Y(i-1) - Y(i)) <= 1e-15)
			{
				return Y(i);
			}

			return ((Y(i)-Y(i-1)) * (xCoord-X(i-1)) / (X(i)-X(i-1))) + Y(i-1);
		}
	}

  // value undefined; consider constant after last curve point
	return Y(numPoints_-1);
}

void adcu::MultiLineCurve::SetPoint( size_t index, real x, real y )
{
	if (index >= numPoints_)
	{
		throw axis::foundation::OutOfBoundsException();
	}
  real *points = points_;
  X(index) = x;
  Y(index) = y;
}

size_t adcu::MultiLineCurve::PointCount( void ) const
{
	return numPoints_;
}

void * adcu::MultiLineCurve::operator new( size_t, void *ptr )
{
  return ptr;
}

void adcu::MultiLineCurve::operator delete( void *, void * )
{
  // nothing to do here
}

afm::RelativePointer adcu::MultiLineCurve::Create( size_t numPoints )
{
  size_t bytes = sizeof(MultiLineCurve);
  afm::RelativePointer ptr = System::ModelMemory().Allocate(bytes);
  new (*ptr) MultiLineCurve(numPoints);
  return ptr;
}

bool adcu::MultiLineCurve::IsGPUCapable( void ) const
{
  return true;
}

afu::Uuid adcu::MultiLineCurve::GetTypeId( void ) const
{
  // FDF72912-AFFB-417A-A414-4760B33C642E
  int uuid[] = {0xFD,0xF7,0x29,0x12,0xAF,0xFB, 0x41,0x7A,0xA4,0x14,0x47,0x60,0xB3,0x3C,0x64,0x2E};
  return afu::Uuid(uuid);
}

adcu::CurveUpdateCommand& adcu::MultiLineCurve::GetUpdateCommand( void )
{
  return multiLineGPUCommand;
}

int adcu::MultiLineCurve::GetGPUDataSize( void ) const
{
  return sizeof(afm::RelativePointer);
}

void adcu::MultiLineCurve::DoInitGPUData( void *data, real *outputBucket )
{
  afm::RelativePointer &ptr = *(afm::RelativePointer *)data;
  ptr = pointArrayPtr_;
}