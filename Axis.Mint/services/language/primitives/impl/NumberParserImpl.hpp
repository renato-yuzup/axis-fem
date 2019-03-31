#pragma once
#include "../ParserImpl.hpp"
#include <boost/spirit/include/qi.hpp>
#include "AxisString.hpp"

namespace axis { namespace services	{	namespace language { namespace primitives	{	namespace impl {

class NumberParserImpl : public axis::services::language::primitives::ParserImpl
{
public:
  NumberParserImpl(void);
  ~NumberParserImpl(void);
private:
	virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
	void InitGrammar(void);

	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _num_rule;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _signal;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _digit_sequence;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _integer;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _decimal_part;
	boost::spirit::qi::rule<axis::services::language::iterators::InputIterator, axis::String()> _scientific_not;
};					

}	}	} } } // namespace axis::services::language::primitives::impl
