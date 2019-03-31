#pragma once
#include "domain/formulations/FormulationStrategy.hpp"
#include "domain/formulations/nlhr_commands/NlhrInternalForceCommand.hpp"
#include "domain/formulations/nlhr_commands/NlhrStrainCommand.hpp"
#include "domain/formulations/nlhr_commands/NlhrUpdateGeometryCommand.hpp"

namespace axis { namespace domain { namespace formulations {

class NonLinearHexaReducedStrategy : public FormulationStrategy
{
public:
  virtual UpdateStrainCommand& GetUpdateStrainCommand( void );
  virtual InternalForceCommand& GetUpdateInternalForceStrategy( void );
  virtual UpdateGeometryCommand& GetUpdateGeometryCommand( void );

  static NonLinearHexaReducedStrategy& GetInstance(void);
private:
  NonLinearHexaReducedStrategy(void);
  virtual ~NonLinearHexaReducedStrategy(void);

  nlhr_commands::NlhrStrainCommand updateStrainCmd_;
  nlhr_commands::NlhrInternalForceCommand internalForceCmd_;
  nlhr_commands::NlhrUpdateGeometryCommand updateGeomCommand_;
  static NonLinearHexaReducedStrategy *instance_;
};

} } } // namespace axis::domain::formulations
