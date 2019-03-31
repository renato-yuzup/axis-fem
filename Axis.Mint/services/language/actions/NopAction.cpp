#include "NopAction.hpp"

namespace asla = axis::services::language::actions;
namespace aslp = axis::services::language::parsing;

void asla::NopAction::Run( const aslp::ParseResult& ) const
{
	/*
		We are a no-op action, so we have nothing to do! :-P
	*/
}

asla::ParserAction& asla::NopAction::Clone( void ) const
{
	return *new NopAction();
}
