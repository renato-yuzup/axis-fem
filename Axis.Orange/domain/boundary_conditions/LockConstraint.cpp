#include "LockConstraint.hpp"
#include "LockConstraintCommand.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

static adbc::LockConstraintCommand lockBcGPUCommand;

namespace {
  struct LockGPUData {
    real ReleaseTime;
  };
}

adbc::LockConstraint::LockConstraint( void ) : BoundaryCondition(Lock)
{
	releaseTime_ = -1;
}

adbc::LockConstraint::LockConstraint( real releaseTime ) : BoundaryCondition(Lock)
{
  releaseTime_ = releaseTime;
}

adbc::LockConstraint::~LockConstraint( void )
{
	releaseTime_ = -1;
}

adbc::BoundaryCondition& adbc::LockConstraint::Clone( void ) const
{
	return *new LockConstraint();
}

void adbc::LockConstraint::Destroy( void ) const
{
	delete this;
}

real adbc::LockConstraint::GetValue( real ) const
{
	return 0;
}

bool adbc::LockConstraint::IsNonNodalLoad( void ) const
{
	return false;
}

bool adbc::LockConstraint::IsVariableConstraint( void ) const
{
	return false;
}

bool adbc::LockConstraint::IsNodalLock( void ) const
{
	return true;
}

bool adbc::LockConstraint::Active( real time ) const
{
  return releaseTime_ < 0 || time <= releaseTime_;
}

bool adbc::LockConstraint::IsGPUCapable( void ) const
{
  return true;
}

afu::Uuid adbc::LockConstraint::GetTypeId( void ) const
{
  // CF00E4E5-6EFD-46B8-B5D2-0BE5883C00B1
  int uuid[] = {0xCF,0x00,0xE4,0xE5,0x6E,0xFD,0x46,0xB8,0xB5,0xD2,0x0B,0xE5,0x88,0x3C,0x00,0xB1};
  return afu::Uuid(uuid);
}

adbc::BoundaryConditionUpdateCommand& adbc::LockConstraint::GetUpdateCommand(void)
{
 return lockBcGPUCommand; 
}

int adbc::LockConstraint::GetGPUDataSize( void ) const
{
  return sizeof(LockGPUData);
}

void axis::domain::boundary_conditions::LockConstraint::InitGPUData( void *data, real& )
{
  LockGPUData *gpuData = (LockGPUData *)data;
  gpuData->ReleaseTime = releaseTime_;
}
