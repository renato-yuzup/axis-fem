#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParameterValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API NullValue : public ParameterValue
{
public:
	NullValue(void);
	~NullValue(void);
	virtual bool IsAssignment(void) const;
	virtual bool IsAtomic(void) const;
	virtual bool IsNull(void) const;
	virtual bool IsArray(void) const;
	virtual axis::String ToString(void) const;
	virtual ParameterValue& Clone(void) const;
};		

} } } } } // namespace axis::services::language::syntax::evaluation
