#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParameterValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API AtomicValue : public ParameterValue
{
public:
	virtual ~AtomicValue(void);
	virtual bool IsAssignment(void) const;
	virtual bool IsAtomic(void) const;
	virtual bool IsNull(void) const;
	virtual bool IsArray(void) const;
	virtual bool IsString(void) const = 0;
	virtual bool IsId(void) const = 0;
	virtual bool IsNumeric(void) const = 0;
};

} } } } } // namespace axis::services::language::syntax::evaluation
