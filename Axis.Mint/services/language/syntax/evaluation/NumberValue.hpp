#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AtomicValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API NumberValue : public AtomicValue
{
public:
	NumberValue(long value);
	NumberValue(double value);
	~NumberValue(void);
	virtual bool IsString(void) const;
	virtual bool IsId(void) const;
	virtual bool IsNumeric(void) const;
	bool IsInteger(void) const;
	long GetLong(void) const;
	double GetDouble(void) const;
	virtual axis::String ToString(void) const;
	virtual ParameterValue& Clone(void) const;
private:
  const long _longValue;
  const double _doubleValue;
  const bool _isInteger;
};	

} } } } } // namespace axis::services::language::syntax::evaluation
