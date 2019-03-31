#include "Curve.hpp"
#include "NullCurveUpdateCommand.hpp"

namespace adcu = axis::domain::curves;
namespace afm = axis::foundation::memory;

adcu::Curve::~Curve( void )
{
	// nothing to do here
}

bool adcu::Curve::IsGPUCapable( void ) const
{
  return false;
}

adcu::CurveUpdateCommand& adcu::Curve::GetUpdateCommand( void )
{
  return NullCurveUpdateCommand::GetInstance();
}

int adcu::Curve::GetGPUDataSize( void ) const
{
  return 0;
}

void adcu::Curve::InitGPUData( void *data, real *outputBucket, const real *readOnlyMirroredOutputBucketAddr)
{
  outputBucket_ = readOnlyMirroredOutputBucketAddr;
  DoInitGPUData(data, outputBucket);
}

const real * adcu::Curve::GetGPUValueSlotPointer( void ) const
{
  return outputBucket_;
}

void adcu::Curve::DoInitGPUData( void *, real * )
{
  // nothing to do in base implementation
}
