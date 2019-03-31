#include "VariableConstraint.hpp"
#include "VariableConstraintCommand.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace adcv = axis::domain::curves;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

namespace {
struct VariableConstraintCPUData
{
  const real * CurveDataPtr;
  real ScalingFactor;
  real ReleaseTime;
};
} // namespace

static adbc::VariableConstraintCommand varBcGPUCommand;

adbc::VariableConstraint::VariableConstraint( ConstraintType type, afm::RelativePointer& history ) :
BoundaryCondition(type), history_(history)
{
	scalar_ = 1.0;
  releaseTime_ = -1;
}

adbc::VariableConstraint::VariableConstraint( ConstraintType type, afm::RelativePointer& history, 
                                              real scalar ) :
BoundaryCondition(type), history_(history)
{
	scalar_ = scalar;
  releaseTime_ = -1;
}

adbc::VariableConstraint::VariableConstraint( ConstraintType type, afm::RelativePointer& history, 
                                             real scalar, real releaseTime ) :
BoundaryCondition(type), history_(history)
{
  scalar_ = scalar;
  releaseTime_ = releaseTime;
}

adbc::VariableConstraint::~VariableConstraint( void )
{
	// intentionally left blank
}

adbc::BoundaryCondition& adbc::VariableConstraint::Clone( void ) const
{
	return *new VariableConstraint(GetType(), const_cast<afm::RelativePointer&>(history_), scalar_);
}

void adbc::VariableConstraint::Destroy( void ) const
{
	delete this;
}

real adbc::VariableConstraint::GetValue( real time ) const
{
	return absref<adcv::Curve>(history_).GetValueAt(time) * scalar_;
}

bool adbc::VariableConstraint::IsNonNodalLoad( void ) const
{
	return false;
}

bool adbc::VariableConstraint::IsVariableConstraint( void ) const
{
	return true;
}

bool adbc::VariableConstraint::IsNodalLock( void ) const
{
	return false;
}

bool adbc::VariableConstraint::Active( real time ) const
{
  return releaseTime_ < 0 || time <= releaseTime_;
}

bool adbc::VariableConstraint::IsGPUCapable( void ) const
{
  return absref<adcv::Curve>(history_).IsGPUCapable();
}

afu::Uuid adbc::VariableConstraint::GetTypeId( void ) const
{
  // 6FAE21A0-D8A0-4E7E-AB83-7AF05B5624CD
  int uuid[] = {0x6F,0xAE,0x21,0xA0,0xD8,0xA0,0x4E,0x7E,0xAB,0x83,0x7A,0xF0,0x5B,0x56,0x24,0xCD};
  return afu::Uuid(uuid);
}

adbc::BoundaryConditionUpdateCommand& 
  adbc::VariableConstraint::GetUpdateCommand(void)
{
  return varBcGPUCommand;
}

int adbc::VariableConstraint::GetGPUDataSize( void ) const
{
 return sizeof(VariableConstraintCPUData);
}

void adbc::VariableConstraint::InitGPUData( void *data, real& outputBucket )
{
  VariableConstraintCPUData *gpuData = (VariableConstraintCPUData *)data;
  adcv::Curve& curve = absref<adcv::Curve>(history_);
  gpuData->CurveDataPtr = curve.GetGPUValueSlotPointer();
  gpuData->ScalingFactor = scalar_;
  gpuData->ReleaseTime = releaseTime_;
}