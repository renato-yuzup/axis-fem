#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "UpdateStrainCommand.hpp"
#include "InternalForceCommand.hpp"
#include "UpdateGeometryCommand.hpp"

namespace axis { namespace domain { namespace formulations {

class AXISCOMMONLIBRARY_API FormulationStrategy
{
public:
  FormulationStrategy(void);
  virtual ~FormulationStrategy(void);
  virtual InternalForceCommand& GetUpdateInternalForceStrategy(void) = 0;
  virtual UpdateStrainCommand& GetUpdateStrainCommand(void) = 0;
  virtual UpdateGeometryCommand& GetUpdateGeometryCommand(void) = 0;

  static FormulationStrategy& NullStrategy;
};

} } } // namespace axis::domain::formulations
