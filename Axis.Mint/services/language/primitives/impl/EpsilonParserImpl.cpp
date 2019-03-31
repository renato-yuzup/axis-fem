#include "EpsilonParserImpl.hpp"
#include "../../parsing/EmptyNode.hpp"

namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::EpsilonParserImpl::EpsilonParserImpl( void )
{
	/* nothing to do here */
}

aslppi::EpsilonParserImpl::~EpsilonParserImpl( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslppi::EpsilonParserImpl::DoParse( const asli::InputIterator& begin, 
                                                      const asli::InputIterator&, bool ) const
{
	// accept, regardless of the input (we are an epsilon production, anyway)
	return aslp::ParseResult(aslp::ParseResult::MatchOk, *new aslp::EmptyNode(), begin);
}
