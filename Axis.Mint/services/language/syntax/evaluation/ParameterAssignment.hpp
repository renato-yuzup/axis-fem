#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParameterValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API ParameterAssignment : public ParameterValue
{
public:
	ParameterAssignment(const axis::String& idName, const ParameterValue& value);
	virtual ~ParameterAssignment(void);
	virtual ParameterValue& Clone(void) const;
	virtual bool IsAssignment(void) const;
	virtual bool IsAtomic(void) const;
	virtual bool IsNull(void) const;
	virtual bool IsArray(void) const;
	axis::String GetIdName(void) const;
	const ParameterValue& GetRhsValue(void) const;
	virtual axis::String ToString(void) const;
private:
	const axis::String _idName;
	const ParameterValue& _value;
};				

} } } } } // namespace axis::services::language::syntax::evaluation
