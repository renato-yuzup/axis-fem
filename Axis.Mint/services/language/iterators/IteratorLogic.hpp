#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace services { namespace language { namespace iterators {

class AXISMINT_API IteratorLogic
{
public:
	IteratorLogic(void);
	virtual ~IteratorLogic(void);

	virtual IteratorLogic& Clone(void) const = 0;

	/* Increment operators */
	virtual IteratorLogic& operator ++(void) = 0;	// pre-fixed
	virtual IteratorLogic& operator ++(int) = 0;	// post-fixed

	virtual bool operator >(const IteratorLogic& other) const = 0;

	/* De-reference operators */
	virtual const axis::String::value_type& operator *(void) const = 0;

	/* Comparison operators */
	virtual bool operator ==(const IteratorLogic& it) const = 0;
	virtual bool operator !=(const IteratorLogic& it) const = 0;

	void NotifyUse(void);
	void NotifyDestroy(void);
private:
  IteratorLogic(const IteratorLogic& other);
  IteratorLogic& operator =(const IteratorLogic& it);

	int _useCount;
};			

} } } } // namespace axis::services::language::iterators
