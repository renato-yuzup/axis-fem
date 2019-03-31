#pragma once
#include "domain/formulations/FormulationStrategy.hpp"
#include "domain/formulations/nlhrfb_commands/NlhrFBInternalForceCommand.hpp"
#include "domain/formulations/nlhrfb_commands/NlhrFBStrainCommand.hpp"
#include "domain/formulations/nlhrfb_commands/NlhrFBUpdateGeometryCommand.hpp"

namespace axis { namespace domain { namespace formulations {

	class NonLinearHexaFBReducedStrategy : public FormulationStrategy
	{
	public:
		virtual UpdateStrainCommand& GetUpdateStrainCommand( void );
		virtual InternalForceCommand& GetUpdateInternalForceStrategy( void );
		virtual UpdateGeometryCommand& GetUpdateGeometryCommand( void );

		static NonLinearHexaFBReducedStrategy& GetInstance(void);
	private:
		NonLinearHexaFBReducedStrategy(void);
		virtual ~NonLinearHexaFBReducedStrategy(void);

		nlhrfb_commands::NlhrFBStrainCommand updateStrainCmd_;
		nlhrfb_commands::NlhrFBInternalForceCommand internalForceCmd_;
		nlhrfb_commands::NlhrFBUpdateGeometryCommand updateGeomCommand_;

		static NonLinearHexaFBReducedStrategy *instance_;
	};

} } } // namespace axis::domain::formulations
