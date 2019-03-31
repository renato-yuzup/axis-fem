#pragma once
#include "domain/formulations/FormulationStrategy.hpp"
#include "lhr_commands/LhrStrainCommand.hpp"
#include "lhr_commands/LhrInternalForceCommand.hpp"
#include "domain/formulations/NullUpdateGeometryCommand.hpp"

namespace axis { namespace domain { namespace formulations {

class LinearHexaReducedStrategy : public FormulationStrategy
{
public:
  LinearHexaReducedStrategy(void);
  virtual ~LinearHexaReducedStrategy(void);
  virtual InternalForceCommand& GetUpdateInternalForceStrategy( void );
  virtual UpdateStrainCommand& GetUpdateStrainCommand( void );
  virtual UpdateGeometryCommand& GetUpdateGeometryCommand( void );

  static LinearHexaReducedStrategy& GetInstance(void);
private:
  lhr_commands::LhrStrainCommand updateStrainCmd_;
  lhr_commands::LhrInternalForceCommand internalForceCmd_;
  NullUpdateGeometryCommand updateGeomCommand_;
  static LinearHexaReducedStrategy *instance_;
};

} } } // namespace axis::domain::formulations
