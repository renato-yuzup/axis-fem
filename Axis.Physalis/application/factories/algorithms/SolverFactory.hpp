#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "foundation/collections/Collectible.hpp"
#include "AxisString.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "domain/algorithms/Solver.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "ClockworkFactory.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "application/locators/ClockworkFactoryLocator.hpp"
#include "domain/algorithms/Clockwork.hpp"

namespace axis { namespace application { namespace factories { namespace algorithms {

class AXISPHYSALIS_API SolverFactory : public axis::foundation::collections::Collectible
{
public:
	SolverFactory(void);
	virtual ~SolverFactory(void);

	virtual bool CanBuild(const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime) const = 0;
	virtual bool CanBuild(const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime, const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& clockworkParams) const = 0;
	virtual axis::domain::algorithms::Solver& BuildSolver(const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime) = 0;
	virtual axis::domain::algorithms::Solver& BuildSolver(const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime, axis::domain::algorithms::Clockwork& clockwork) = 0;

	/**********************************************************************************************//**
		* <summary> Destroys this object.</summary>
		**************************************************************************************************/
	virtual void Destroy(void) const = 0;
};			

} } } } // namespace axis::application::factories::algorithms
