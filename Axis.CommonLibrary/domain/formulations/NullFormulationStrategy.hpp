#pragma once
#include "FormulationStrategy.hpp"
#include "NullUpdateStrainCommand.hpp"
#include "NullInternalForceCommand.hpp"
#include "NullUpdateGeometryCommand.hpp"

namespace axis { namespace domain { namespace formulations {

class NullFormulationStrategy : public FormulationStrategy
{
public:
  NullFormulationStrategy(void);
  ~NullFormulationStrategy(void);
  virtual InternalForceCommand& GetUpdateInternalForceStrategy( void );
  virtual UpdateStrainCommand& GetUpdateStrainCommand( void );
  virtual UpdateGeometryCommand& GetUpdateGeometryCommand(void);
private:
  NullUpdateStrainCommand strainCommand_;
  NullInternalForceCommand internalForceCommand_;
  NullUpdateGeometryCommand updateGeomCommand_;
};

} } } // namespace axis::domain::formulations
