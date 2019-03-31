#pragma once
#include "../ParserImpl.hpp"
#include <boost/spirit/include/qi.hpp>
#include "AxisString.hpp"

namespace axis { namespace services	{	namespace language { namespace primitives	{	namespace impl {

class EoiParserImpl : public axis::services::language::primitives::ParserImpl
{
public:
  EoiParserImpl(void);
  ~EoiParserImpl(void);
protected:
	virtual axis::services::language::parsing::ParseResult DoParse(const axis::services::language::iterators::InputIterator& begin, const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
};					

}	}	}	} }

