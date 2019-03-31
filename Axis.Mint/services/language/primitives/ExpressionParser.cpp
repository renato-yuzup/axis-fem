#include "ExpressionParser.hpp"
#include "../actions/ChainedAction.hpp"
#include "../actions/NopAction.hpp"

namespace asla = axis::services::language::actions;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::ExpressionParser::ExpressionParser( void )
{
	_action = new actions::NopAction();
}

aslpp::ExpressionParser::~ExpressionParser( void )
{
	delete _action;
}

aslp::ParseResult aslpp::ExpressionParser::Parse( const asli::InputIterator& begin, 
                                                  const asli::InputIterator& end, 
                                                  bool trimSpaces /*= true*/ ) const
{
	aslp::ParseResult result = DoParse(begin, end, trimSpaces);
	if (result.GetResult() == aslp::ParseResult::MatchOk)
	{	// execute action
		_action->Run(result);
	}
	return result;
}

aslp::ParseResult aslpp::ExpressionParser::operator()( const asli::InputIterator& begin, 
                                                       const asli::InputIterator& end, 
                                                       bool trimSpaces /*= true*/ ) const
{
	return Parse(begin, end, trimSpaces);
}

aslpp::ExpressionParser& aslpp::ExpressionParser::AddAction( const asla::ParserAction& action )
{
	// clone and chain current action
	asla::ParserAction *newAction = new asla::ChainedAction(*_action, action);	
	// delete and replace old action
	delete _action;
	_action = newAction;
	return *this;
}

aslpp::ExpressionParser& aslpp::ExpressionParser::operator<<( const asla::ParserAction& action )
{
	return AddAction(action);
}
