#include "AnyParserImpl.hpp"
#include "foundation/ApplicationErrorException.hpp"
#include "../../parsing/OperatorTerminal.hpp"
#include "../../parsing/ReservedWordTerminal.hpp"
#include "../../parsing/EmptyNode.hpp"

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
//#include <boost/spirit/home/phoenix/container.hpp>
//#include <boost/spirit/home/phoenix/bind/bind_member_function.hpp>
//#include <boost/spirit/home/phoenix/statement/sequence.hpp>

namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::AnyParserImpl::AnyParserImpl( const axis::String& expectedValue, 
                                      const ResultType resultType, int associatedValue /*= 0*/, 
                                      int precedence /*= 0*/, int associativity /*= 0*/ ) :
expectedValue_(expectedValue), resultType_(resultType), associatedValue_(associatedValue), 
precedence_(precedence), associativity_(associativity)
{
	/* nothing to do here */
}

aslppi::AnyParserImpl::~AnyParserImpl( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslppi::AnyParserImpl::DoParse( const asli::InputIterator& begin, 
                                                  const asli::InputIterator& end, 
                                                  bool trimSpaces /*= true*/ ) const
{
	asli::InputIterator it(begin);
	String value;

	aslp::ParseResult::Result result;
	aslp::ParseTreeNode *node;

	// try to remove whitespaces before, if asked
	if(trimSpaces) SkipWhitespaces(it, end);

	size_t pos = 0;
	asli::InputIterator parseStartPos(it);
	bool ret = true;
	while (pos < expectedValue_.size() && it != end)
	{
		axis::String::value_type c = *it;
		if (expectedValue_[pos] != c)
		{	// failed match
			ret = false;
			it = parseStartPos;
			break;
		}
		value += c;
		++pos;
		++it;
	}	
	if (pos != expectedValue_.size()) 
	{	// we couldn't reach the end of the patter due to insufficient chars in input stream
		ret = false;
	}

	if (ret && it != end)
	{	// even if we succeed, we must make sure that we didn't take part
		// of a declaration of an operator/reserved word; for example, if 
		// we are expecting an operator named 'OR', we should not accept
		// 'ORELSE' neither 'OR123' -- however 'OR=1', 'OR(XYZ)' and
		// 'OR ELSE' are acceptable
		
		// check next char and compare its char class with the last
		// char of our token
		CharType myLastChar = CheckCharType(*--expectedValue_.cend());
		CharType nextRemainingChar = CheckCharType(*it);

		ret = 
      (myLastChar == kIdentifierDeclarationType && nextRemainingChar == kNonIdentifierDeclarationType) 
      || myLastChar == kNonIdentifierDeclarationType;
	}

	// try to remove whitespaces after
	if(trimSpaces) SkipWhitespaces(it, end);

	if (ret)	// full match
	{	
		switch (resultType_)
		{
		case kOperator:			/* must output an operator */
			node = new aslp::OperatorTerminal(value, associatedValue_, precedence_, associativity_);
			break;
		case kReservedWord:		/* must output a reserved word */
			node = new aslp::ReservedWordTerminal(value, associatedValue_);
			break;
		default:				/* huh? something is wrong here... */
			throw axis::foundation::ApplicationErrorException(
        _T("Internal error: unexpected specified output type value when parsing using AnyParser."));
		}
		result = aslp::ParseResult::MatchOk;
	}
	else
	{
		result = aslp::ParseResult::FailedMatch;
		node = new aslp::EmptyNode();

		// force go back to the beginning
		it = begin;
	}
	return aslp::ParseResult(result, *node, it);
}

aslppi::AnyParserImpl::CharType aslppi::AnyParserImpl::CheckCharType( axis::String::value_type c ) const
{
	if ((c >= (axis::String::value_type)'A' && c <= (axis::String::value_type)'Z') ||
		(c >= (axis::String::value_type)'a' && c <= (axis::String::value_type)'z') ||
		(c >= (axis::String::value_type)'0' && c <= (axis::String::value_type)'9') ||
		(c == (axis::String::value_type)'_'))
	{	
		return kIdentifierDeclarationType;
	}
	return kNonIdentifierDeclarationType;
}

void aslppi::AnyParserImpl::SkipWhitespaces( asli::InputIterator& it, 
                                             const asli::InputIterator& end ) const
{
	axis::String::value_type c = *it;
	while ((c == _T('\t') || c == _T(' ') || c == _T('\n') || c == _T('\r') || c == _T('\0')) && it != end)
	{
		++it;
		c = *it;
	}
}