#include "Parser.hpp"
#include "ParserImpl.hpp"

namespace asla = axis::services::language::actions;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::Parser::Parser( const Parser& other )
{
	parserImpl_ = NULL;
	lastParserImplInChain_ = NULL;
	Copy(other);
}

aslpp::Parser::Parser( ParserImpl& implementationLogic )
{
	// increment reference counter
	parserImpl_ = &implementationLogic;
	lastParserImplInChain_ = parserImpl_;
	implementationLogic.NotifyUse();
}

aslpp::Parser::Parser( ParserImpl *firstParserInChain, ParserImpl *lastParserInChain )
{
	parserImpl_ = firstParserInChain;
	lastParserImplInChain_ = lastParserInChain;
	parserImpl_->NotifyUse();
}

aslpp::Parser::~Parser( void )
{
	// decrement reference counter
	parserImpl_->NotifyDestroy();

	/*
		NOTE: we are not notifying other parsers to destroy because that will
		be done by the first parser in chain
	*/
}

aslp::ParseResult aslpp::Parser::Parse( const asli::InputIterator& begin, 
                                        const asli::InputIterator& end, bool trimSpaces /*= true*/ ) const
{
	return parserImpl_->Parse(begin, end, trimSpaces);
}

aslpp::Parser& aslpp::Parser::operator=( const Parser& other )
{
	Copy(other);
	return *this;
}

void aslpp::Parser::Copy( const Parser& other )
{
	if (other.parserImpl_ == this->parserImpl_)
	{	// uses the same business object, ignore it
		return;
	}	

	if (parserImpl_ != NULL)
	{
		parserImpl_->NotifyDestroy();
	}
	parserImpl_ = other.parserImpl_;
	parserImpl_->NotifyUse();
	lastParserImplInChain_ = other.lastParserImplInChain_;
}

aslpp::Parser& aslpp::Parser::AddAction( const asla::ParserAction& action )
{
	parserImpl_->AddAction(action);
	return *this;
}

aslpp::Parser& aslpp::Parser::operator<<( const asla::ParserAction& action )
{
	return AddAction(action);
}

aslp::ParseResult aslpp::Parser::operator()( const asli::InputIterator& begin, 
                                             const asli::InputIterator& end, 
                                             bool trimSpaces /*= true*/ ) const
{
	return Parse(begin, end, trimSpaces);
}
