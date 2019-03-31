#pragma once
#include "IteratorLogic.hpp"
#include "AxisString.hpp"

namespace axis { namespace services { namespace language { namespace iterators {

class StringIteratorLogic : public IteratorLogic
{
public:
	StringIteratorLogic(const StringIteratorLogic& it);
	StringIteratorLogic(const axis::String& sourceStr);
	StringIteratorLogic(const axis::String::const_iterator& sourceIt);

	~StringIteratorLogic(void);

	virtual IteratorLogic& Clone(void) const;

	/* Increment operators */
	virtual IteratorLogic& operator ++(void);	// pre-fixed
	virtual IteratorLogic& operator ++(int);	// post-fixed

	virtual bool operator >(const IteratorLogic& other) const;

	/* De-reference operators */
	virtual const axis::String::value_type& operator *(void) const;

	/* Comparison operators */
	virtual bool operator ==(const IteratorLogic& it) const;
	virtual bool operator !=(const IteratorLogic& it) const;
private:
	axis::String::iterator _it;
};			

} } } } // namespace axis::services::language::iterators
