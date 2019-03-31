#include "StringParserImpl.hpp"

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
//#include <boost/spirit/home/phoenix/container.hpp>
//#include <boost/spirit/home/phoenix/bind/bind_member_function.hpp>
//#include <boost/spirit/home/phoenix/statement/sequence.hpp>
#include "../../parsing/StringTerminal.hpp"
#include "../../parsing/EmptyNode.hpp"


namespace bs = boost::spirit;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::StringParserImpl::StringParserImpl( bool escapedString )
{
	// initialize grammar
	if (escapedString)
	{
		_char_rule = bs::qi::lexeme[_T("\\\\")][bs::qi::_val = _T("\\")] | 
                 bs::qi::lexeme[_T("\\\"")][bs::qi::_val = _T("\"")] | 
                 (bs::qi::char_ - _T("\"") - _T("\\"))[bs::qi::_val = bs::qi::_1];
	}
	else
	{
		_char_rule = (bs::qi::char_ - _T("\""));
	}
	_string_rule = (bs::qi::lexeme[_T("\"")] >> 
                 *_char_rule[bs::qi::_val += bs::qi::_1] >> 
                 bs::qi::lexeme[_T("\"")]);
}

aslppi::StringParserImpl::~StringParserImpl( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslppi::StringParserImpl::DoParse( const asli::InputIterator& begin, 
                                                     const asli::InputIterator& end, 
                                                     bool trimSpaces /*= true*/ ) const
{
	asli::InputIterator it(begin);
	String value;
	// remove whitespaces before, if asked
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank);	
	bool ret = bs::qi::parse(it, end, _string_rule, value);
	// remove whitespaces at the end, if asked
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank);	
	if (ret)
	{
		return aslp::ParseResult(aslp::ParseResult::MatchOk, *new aslp::StringTerminal(value), it);
	}
	return aslp::ParseResult(aslp::ParseResult::FailedMatch, *new aslp::EmptyNode(), it);
}
