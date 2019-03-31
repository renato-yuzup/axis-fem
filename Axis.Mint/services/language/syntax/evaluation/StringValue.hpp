#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AtomicValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API StringValue : public AtomicValue
{
public:
	StringValue(const axis::String& value);
	~StringValue(void);
	virtual bool IsString(void) const;
	virtual bool IsId(void) const;
	virtual bool IsNumeric(void) const;
	axis::String GetValue(void) const;
	virtual axis::String ToString(void) const;
	virtual ParameterValue& Clone(void) const;
private:
  const axis::String _value;
};				

} } } } } // namespace axis::services::language::syntax::evaluation