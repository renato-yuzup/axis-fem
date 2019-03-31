#pragma once
#include "../ParserImpl.hpp"
#include <boost/spirit/include/qi.hpp>
#include "AxisString.hpp"

namespace axis { namespace services	{	namespace language { namespace primitives	{	namespace impl {

class IdParserImpl : public axis::services::language::primitives::ParserImpl
{
public:
	IdParserImpl(void);
	~IdParserImpl(void);
private:
	virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;

	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _id_rule;
};

}	}	}	} } // namespace axis::services::language::primitives::impl
