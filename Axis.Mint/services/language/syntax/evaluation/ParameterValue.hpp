#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {
class AXISMINT_API ParameterValue
{
public:
	virtual ~ParameterValue(void);

	virtual bool IsAssignment(void) const = 0;
	virtual bool IsAtomic(void) const = 0;
	virtual bool IsNull(void) const = 0;
	virtual bool IsArray(void) const = 0;

	virtual axis::String ToString(void) const = 0;

	virtual ParameterValue& Clone(void) const = 0;
};				

} } } } } // namespace axis::services::language::syntax::evaluation
