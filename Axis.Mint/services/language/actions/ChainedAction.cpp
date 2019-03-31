#include "ChainedAction.hpp"

namespace asla = axis::services::language::actions;
namespace aslp = axis::services::language::parsing;

asla::ChainedAction::ChainedAction( const ParserAction& myAction, const ParserAction& nextAction )
{
	_nextAction = &nextAction.Clone();
	_myAction = &myAction.Clone();
}

asla::ChainedAction::~ChainedAction( void )
{
	delete _myAction;
	delete _nextAction;
}

void asla::ChainedAction::Run( const aslp::ParseResult& result ) const
{
	_myAction->Run(result);
	_nextAction->Run(result);
}

asla::ParserAction& asla::ChainedAction::Clone( void ) const
{
	return *new ChainedAction(*_myAction, *_nextAction);
}