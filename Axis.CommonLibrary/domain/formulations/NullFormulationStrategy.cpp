#include "NullFormulationStrategy.hpp"

namespace adf = axis::domain::formulations;

adf::NullFormulationStrategy::NullFormulationStrategy(void)
{
  // nothing to do here
}

adf::NullFormulationStrategy::~NullFormulationStrategy(void)
{
  // nothing to do here
}

adf::InternalForceCommand&
  adf::NullFormulationStrategy::GetUpdateInternalForceStrategy( void )
{
  return internalForceCommand_;
}

adf::UpdateStrainCommand& 
  adf::NullFormulationStrategy::GetUpdateStrainCommand( void )
{
  return strainCommand_;
}

adf::UpdateGeometryCommand& 
  adf::NullFormulationStrategy::GetUpdateGeometryCommand( void )
{
  return updateGeomCommand_;
}
