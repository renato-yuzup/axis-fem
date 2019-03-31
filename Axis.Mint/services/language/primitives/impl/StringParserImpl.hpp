#pragma once
#include "../ParserImpl.hpp"
#include <boost/spirit/include/qi.hpp>
#include "AxisString.hpp"

namespace axis { namespace services	{	namespace language { namespace primitives	{	namespace impl {

class StringParserImpl : public axis::services::language::primitives::ParserImpl
{
public:
  StringParserImpl(bool escapedString);
  ~StringParserImpl(void);
private:
	virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _string_rule;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _char_rule;
};					

}	}	}	} } // namespace axis::services::language::primitives::impl
