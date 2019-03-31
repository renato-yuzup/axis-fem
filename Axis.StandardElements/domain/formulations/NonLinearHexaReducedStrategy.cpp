#include "stdafx.h"
#include "NonLinearHexaReducedStrategy.hpp"

namespace adf = axis::domain::formulations;
namespace adfn = axis::domain::formulations::nlhr_commands;

adf::NonLinearHexaReducedStrategy * adf::NonLinearHexaReducedStrategy
  ::instance_ = new NonLinearHexaReducedStrategy();

adf::NonLinearHexaReducedStrategy::NonLinearHexaReducedStrategy(void)
{
  // nothing to do here
}

adf::NonLinearHexaReducedStrategy::~NonLinearHexaReducedStrategy(void)
{
  // nothing to do here
}

adf::NonLinearHexaReducedStrategy& 
  adf::NonLinearHexaReducedStrategy::GetInstance( void )
{
  return *instance_;
}

adf::UpdateStrainCommand& 
  adf::NonLinearHexaReducedStrategy::GetUpdateStrainCommand( void )
{
  return updateStrainCmd_;
}

adf::InternalForceCommand& 
  adf::NonLinearHexaReducedStrategy::GetUpdateInternalForceStrategy( void )
{
  return internalForceCmd_;
}

adf::UpdateGeometryCommand& 
  adf::NonLinearHexaReducedStrategy::GetUpdateGeometryCommand( void )
{
  return updateGeomCommand_;
}
