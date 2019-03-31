#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "foundation/collections/Collectible.hpp"
#include "AxisString.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "domain/algorithms/Clockwork.hpp"

namespace axis { namespace application { namespace factories { namespace algorithms {

class AXISPHYSALIS_API ClockworkFactory : public axis::foundation::collections::Collectible
{
public:
	virtual ~ClockworkFactory(void);

	virtual void Destroy(void) const = 0;
	virtual bool CanBuild(const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime) const = 0;
	virtual axis::domain::algorithms::Clockwork& Build(const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime) = 0;
};

} } } } // namespace axis::application::factories::algorithms
