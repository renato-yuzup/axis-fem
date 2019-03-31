#include "BlankParserImpl.hpp"
#include "services/language/parsing/EmptyNode.hpp"

namespace bs = boost::spirit;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::BlankParserImpl::BlankParserImpl( bool requiredSpace ) : _requiredSpace(requiredSpace)
{
	// init grammar
	_blank_rule %= *bs::qi::blank | bs::qi::eps;
}

aslppi::BlankParserImpl::~BlankParserImpl( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslppi::BlankParserImpl::DoParse( const asli::InputIterator& begin, 
                                                    const asli::InputIterator& end, bool ) const
{
	asli::InputIterator it(begin);
	String dummy;
	bool result = bs::qi::parse(it, end, _blank_rule, dummy);
	
	// if any space is required, succeed only we we advanced our iterator
	result = _requiredSpace? it != begin : result;

	if (result)
	{	// success!
		return aslp::ParseResult(aslp::ParseResult::MatchOk, *new aslp::EmptyNode(), it);
	}

	// failed
	return aslp::ParseResult(aslp::ParseResult::FailedMatch, *new aslp::EmptyNode(), it);
}