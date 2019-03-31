#include "EoiParserImpl.hpp"
#include "services/language/parsing/EmptyNode.hpp"
#include <boost/spirit/include/qi.hpp>

namespace bs = boost::spirit;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::EoiParserImpl::EoiParserImpl( void )
{
	// nothing to do here
}

aslppi::EoiParserImpl::~EoiParserImpl( void )
{
	// nothing to do here
}
aslp::ParseResult aslppi::EoiParserImpl::DoParse( const asli::InputIterator& begin, 
                                                  const asli::InputIterator& end, 
                                                  bool trimSpaces /*= true*/ ) const
{
	asli::InputIterator it(begin);
	axis::String value;

	// try to remove whitespaces before, if asked
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank);

	// success only if we reached the end of the input
	return aslp::ParseResult((it == end)? aslp::ParseResult::MatchOk : aslp::ParseResult::FailedMatch, 
    *new aslp::EmptyNode(), it);
}
