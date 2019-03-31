#include "stdafx.h"
#include "LinearHexaReducedStrategy.hpp"

namespace adf = axis::domain::formulations;
namespace adflhr = axis::domain::formulations::lhr_commands;

adf::LinearHexaReducedStrategy * adf::LinearHexaReducedStrategy::instance_ = 
  new adf::LinearHexaReducedStrategy();

adf::LinearHexaReducedStrategy::LinearHexaReducedStrategy(void)
{
  // nothing to do here
}

adf::LinearHexaReducedStrategy::~LinearHexaReducedStrategy(void)
{
  // nothing to do here
}

adf::InternalForceCommand&
  adf::LinearHexaReducedStrategy::GetUpdateInternalForceStrategy( void )
{
  return internalForceCmd_;
}

adf::UpdateStrainCommand&
  adf::LinearHexaReducedStrategy::GetUpdateStrainCommand( void )
{
  return updateStrainCmd_;
}

adf::UpdateGeometryCommand& 
  adf::LinearHexaReducedStrategy::GetUpdateGeometryCommand( void )
{
  return updateGeomCommand_;
}

adf::LinearHexaReducedStrategy& adf::LinearHexaReducedStrategy::GetInstance( void )
{
  return *instance_;
}
