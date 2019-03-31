#include "AxisGrammar.hpp"
#include "../primitives/Parser.hpp"
#include "../primitives/impl/IdParserImpl.hpp"
#include "../primitives/impl/NumberParserImpl.hpp"
#include "../primitives/impl/BlankParserImpl.hpp"
#include "../primitives/impl/StringParserImpl.hpp"
#include "../primitives/impl/AnyParserImpl.hpp"
#include "../primitives/impl/EpsilonParserImpl.hpp"
#include "../primitives/impl/EoiParserImpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "../grammar_tokens.hpp"
#include "../parsing/ParseResult.hpp"
#include "IteratorFactory.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslpp = axis::services::language::primitives;
namespace aslp = axis::services::language::parsing;

aslf::AxisGrammar::AxisGrammar( void )
{
	// nothing to do
}

aslpp::Parser aslf::AxisGrammar::CreateIdParser( void )
{
	return aslpp::Parser(*new aslpp::impl::IdParserImpl());
}

aslpp::Parser aslf::AxisGrammar::CreateNumberParser( void )
{
	return aslpp::Parser(*new aslpp::impl::NumberParserImpl());
}

aslpp::Parser aslf::AxisGrammar::CreateBlankParser( bool enforceSpace /*= false */ )
{
	return aslpp::Parser(*new aslpp::impl::BlankParserImpl(enforceSpace));
}

aslpp::Parser aslf::AxisGrammar::CreateStringParser( bool acceptEscapedSequence /*= false*/ )
{
	return aslpp::Parser(*new aslpp::impl::StringParserImpl(acceptEscapedSequence));
}

aslpp::Parser aslf::AxisGrammar::CreateOperatorParser( const axis::String& operatorSpelling, 
                                                       int associatedValue /*= 0 */, 
                                                       int precedence /*= 0*/, 
                                                       int associativity /*= 0*/ )
{
	// check that this is not a reserved keyword
	if (AxisGrammar::IsReservedWord(operatorSpelling))
	{
		throw axis::foundation::ArgumentException();
	}
	return aslpp::Parser(*new aslpp::impl::AnyParserImpl(operatorSpelling, 
    aslpp::impl::AnyParserImpl::kOperator, associatedValue, precedence, associativity));
}

aslpp::Parser aslf::AxisGrammar::CreateReservedWordParser( const axis::String& operatorSpelling, 
                                                           int associatedValue /*= 0 */ )
{
	return aslpp::Parser(*new aslpp::impl::AnyParserImpl(operatorSpelling, 
    aslpp::impl::AnyParserImpl::kReservedWord, associatedValue));
}

aslpp::Parser aslf::AxisGrammar::CreateEpsilonParser( void )
{	
	return aslpp::Parser(*new aslpp::impl::EpsilonParserImpl());
}

aslpp::Parser aslf::AxisGrammar::CreateEoiParser( void )
{
	return aslpp::Parser(*new aslpp::impl::EoiParserImpl());
}

bool aslf::AxisGrammar::IsReservedWord( const axis::String& word )
{
	return (
			word.compare(AXIS_GRAMMAR_BLOCK_OPEN_DELIMITER) == 0 ||
			word.compare(AXIS_GRAMMAR_BLOCK_CLOSE_DELIMITER) == 0 ||
			word.compare(AXIS_GRAMMAR_PARAMETER_START) == 0 
		   );
}

aslpp::Parser aslf::AxisGrammar::CreateReservedWordBlockHeadParser( void )
{
	return aslpp::Parser(*new aslpp::impl::AnyParserImpl(AXIS_GRAMMAR_BLOCK_OPEN_DELIMITER, 
    aslpp::impl::AnyParserImpl::kReservedWord, 0));
}

aslpp::Parser aslf::AxisGrammar::CreateReservedWordBlockTailParser( void )
{
	return aslpp::Parser(*new aslpp::impl::AnyParserImpl(AXIS_GRAMMAR_BLOCK_CLOSE_DELIMITER, 
    aslpp::impl::AnyParserImpl::kReservedWord, 0));
}

aslpp::Parser aslf::AxisGrammar::CreateReservedWordParameterListStarterParser( void )
{
	return aslpp::Parser(*new aslpp::impl::AnyParserImpl(AXIS_GRAMMAR_PARAMETER_START, 
    aslpp::impl::AnyParserImpl::kReservedWord, 0));
}

asli::InputIterator aslf::AxisGrammar::ExtractNextToken( const asli::InputIterator& begin, 
                                                         const asli::InputIterator& end )
{
	asli::InputIterator start = begin;
	aslp::ParseResult result;

	// scan stream for every recognized symbol in the grammar
	// note that each set of tokens is mutually exclusive
	
	while(!result.IsMatch() && start != end)
	{
		asli::InputIterator pos;
		pos = ExtractNextOperator(start, end);
		if (pos != begin) return pos;
	
		pos = ExtractNextReservedWord(start, end);
		if (pos != begin) return pos;

		// not an operator nor keyword -- check for other primitive constructs
		aslpp::Parser parser = CreateIdParser();
		result = parser.Parse(start, end);
		if (result.IsMatch()) return result.GetLastReadPosition();

		parser = CreateNumberParser();
		result = parser.Parse(start, end);
		if (result.IsMatch()) return result.GetLastReadPosition();

		parser = CreateStringParser();
		result = parser.Parse(start, end);
		if (result.IsMatch()) return result.GetLastReadPosition();

		// no matches, possibly an invalid symbol sequence; drop next char until 
		// we can parse something 
		++start;			
	}

	return start;
}

asli::InputIterator aslf::AxisGrammar::ExtractNextOperator( const asli::InputIterator& begin, 
                                                            const asli::InputIterator& end )
{
	aslp::ParseResult result;

	aslpp::Parser parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_AND);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_OR);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_NOT);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_SUM);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_SUBTRACTION);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_PRODUCT);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_DIVIDE);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateOperatorParser(AXIS_GRAMMAR_OPERATOR_REMAINDER);
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	// no matches
	return begin;
}

asli::InputIterator aslf::AxisGrammar::ExtractNextReservedWord( const asli::InputIterator& begin, 
                                                                const asli::InputIterator& end )
{
	aslp::ParseResult result;

	aslpp::Parser parser = CreateReservedWordBlockHeadParser();
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateReservedWordBlockTailParser();
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	parser = CreateReservedWordParameterListStarterParser();
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return result.GetLastReadPosition();

	// no matches
	return begin;
}

bool aslf::AxisGrammar::IsValidToken( const axis::String& symbol )
{
	asli::InputIterator begin = IteratorFactory::CreateStringIterator(symbol);
	asli::InputIterator end = IteratorFactory::CreateStringIterator(symbol.end());
	aslp::ParseResult result;

	// try to extract (or parse) symbol

	asli::InputIterator pos;
	pos = ExtractNextOperator(begin, end);
	if (pos != begin) return true;

	pos = ExtractNextReservedWord(begin, end);
	if (pos != begin) return true;

	// not an operator nor keyword -- check for other primitive constructs
	aslpp::Parser parser = CreateIdParser();
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return true;

	parser = CreateNumberParser();
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return true;

	parser = CreateStringParser();
	result = parser.Parse(begin, end);
	if (result.IsMatch()) return true;

	// no matches
	return false;
}