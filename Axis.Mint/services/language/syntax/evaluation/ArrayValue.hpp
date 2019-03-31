#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParameterValue.hpp"
#include "ArrayValueList.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API ArrayValue : public ParameterValue
{
public:
	ArrayValue(void);
	~ArrayValue(void);

	virtual bool IsAssignment(void) const;
	virtual bool IsAtomic(void) const;
	virtual bool IsNull(void) const;
	virtual bool IsArray(void) const;
	bool IsEmpty(void) const;
	int Count(void) const;

	ArrayValueList::Iterator begin(void) const;
	ArrayValueList::Iterator end(void) const;
	ParameterValue& Get(int pos) const;
	ParameterValue& operator[](int pos) const;
	ArrayValue& AddValue(ParameterValue& value);
	void Clear(void);

	virtual axis::String ToString(void) const;
	virtual ParameterValue& Clone(void) const;
private:
	ArrayValueList& _items;
};			

} } } } } // namespace axis::services::language::syntax::evaluation
