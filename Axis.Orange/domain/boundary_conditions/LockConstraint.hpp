#pragma once
#include "domain/boundary_conditions/BoundaryCondition.hpp"

namespace axis { namespace domain { namespace boundary_conditions {

class LockConstraint : public BoundaryCondition
{
public:
	LockConstraint(void);
  LockConstraint(real releaseTime);
	~LockConstraint(void);
	virtual BoundaryCondition& Clone(void) const;
	virtual void Destroy( void ) const;
	virtual real GetValue( real time ) const;
	virtual bool IsNonNodalLoad( void ) const;
	virtual bool IsVariableConstraint( void ) const;
	virtual bool IsNodalLock( void ) const;
  virtual bool Active(real time) const;
  virtual bool IsGPUCapable( void ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
  virtual BoundaryConditionUpdateCommand& GetUpdateCommand( void );
  virtual int GetGPUDataSize( void ) const;
  virtual void InitGPUData( void *data, real& outputBucket );
private:
  real releaseTime_;
};

} } } // namespace axis::domain::boundary_conditions
