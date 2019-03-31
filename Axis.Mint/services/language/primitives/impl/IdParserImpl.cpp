#include "IdParserImpl.hpp"
#include <boost/spirit/include/phoenix_stl.hpp>
#include "../../parsing/IdTerminal.hpp"
#include "../../parsing/EmptyNode.hpp"
#include "../../factories/AxisGrammar.hpp"
#include "../../factories/IteratorFactory.hpp"

namespace bs = boost::spirit;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::IdParserImpl::IdParserImpl( void )
{
	// initialize grammar
	_id_rule = 
    bs::qi::lexeme[
      (bs::qi::char_('_') | bs::qi::alpha)[boost::phoenix::push_back(bs::qi::_val, bs::qi::_1)] 
        >> *(bs::qi::char_('_') | bs::qi::alnum)[boost::phoenix::push_back(bs::qi::_val, bs::qi::_1)]
    ];
}

aslppi::IdParserImpl::~IdParserImpl( void )
{
	// nothing to do here
}

aslp::ParseResult aslppi::IdParserImpl::DoParse( const asli::InputIterator& begin, 
                                                 const asli::InputIterator& end, 
                                                 bool trimSpaces /*= true*/ ) const
{
	asli::InputIterator it(begin);
	axis::String value;

	bs::qi::rule<asli::InputIterator, axis::String()> _my_rule;
	_my_rule %= bs::qi::alpha >> bs::qi::alpha;
	axis::String teste = _T("abc");
	axis::String val;
	asli::InputIterator mit = aslf::IteratorFactory::CreateStringIterator(teste);
	const asli::InputIterator mend = aslf::IteratorFactory::CreateStringIterator(teste.end());
	bool r = bs::qi::parse(mit, mend, _my_rule, val);

	// try to remove whitespaces before, if asked
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank);
	asli::InputIterator idStart(it);

	bool ret = bs::qi::parse(it, end, _id_rule, value);
	if (aslf::AxisGrammar::IsReservedWord(value))
	{	// cannot use a reserved word as an ID
		return aslp::ParseResult(aslp::ParseResult::FailedMatch, *new aslp::EmptyNode(), idStart);
	}

	// try to remove whitespaces after, if asked
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank);

	if (ret)
	{
		return aslp::ParseResult(aslp::ParseResult::MatchOk, *new aslp::IdTerminal(value), it);
	}
	return aslp::ParseResult(aslp::ParseResult::FailedMatch, *new aslp::EmptyNode(), it);
}
