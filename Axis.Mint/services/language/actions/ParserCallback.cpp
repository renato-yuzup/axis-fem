#include "ParserCallback.hpp"

namespace asla = axis::services::language::actions;
namespace aslp = axis::services::language::parsing;

asla::ParserCallback::~ParserCallback( void )
{
	// default implementation -- do nothing
}

void asla::ParserCallback::ProcessLexerSuccessEvent( const aslp::ParseResult& )
{
	// default implementation -- do nothing
}

void asla::ParserCallback::ProcessLexerSuccessEvent( const aslp::ParseResult&, int )
{
	// default implementation -- do nothing
}

void asla::ParserCallback::ProcessLexerSuccessEvent( const aslp::ParseResult&, void * )
{
	// default implementation -- do nothing
}