#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"
#include "../iterators/InputIterator.hpp"

namespace axis { namespace services { namespace language { namespace factories {

class AXISMINT_API IteratorFactory
{
private:
	IteratorFactory(void);	/* This class cannot be instantiated */
public:
	static axis::services::language::iterators::InputIterator CreateStringIterator(
    const axis::String& inputString);
	static axis::services::language::iterators::InputIterator CreateStringIterator(
    const axis::String::const_iterator& sourceIterator);
};

} } } } // namespace axis::services::language::factories
