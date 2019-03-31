#include "BlockParser.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "../error_messages.hpp"
#include "services/logging/event_sources.hpp"
#include "services/messaging/ErrorMessage.hpp"

namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aaj = axis::application::jobs;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;

aapps::BlockParser::BlockParser( void )
{
}

aapps::BlockParser::~BlockParser(void)
{
	// no op
}

void aapps::BlockParser::SetAnalysis( aaj::StructuralAnalysis& a )
{
	analysis_ = &a;
}

aaj::StructuralAnalysis& aapps::BlockParser::GetAnalysis( void ) const
{
	if (analysis_ == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}

	return *analysis_;
}

void aapps::BlockParser::DetachFromAnalysis( void )
{
	analysis_ = NULL;
}

void aapps::BlockParser::StartContext( aapc::ParseContext& context )
{
	// load attributes
	parseContext_ = &context;

	// call overloaded function
	DoStartContext();
}

void aapps::BlockParser::CloseContext( void )
{
	// just call overloaded function
	DoCloseContext();
}

void aapps::BlockParser::DoCloseContext( void )
{
	// nothing to do in base implementation
}

void aapps::BlockParser::DoStartContext( void )
{
	// nothing to do in base implementation
}

aapc::ParseContext& aapps::BlockParser::GetParseContext( void ) const
{
	return *parseContext_;
}

void aapps::BlockParser::WarnUnrecognizedParams( const aslse::ParameterList& paramList ) const
{
	aslse::ParameterList::Iterator end = paramList.end();
	for (aslse::ParameterList::Iterator it = paramList.begin(); it != end; ++it)
	{
		axis::String s = AXIS_ERROR_MSG_UNKNOWN_BLOCK_PARAM;
		s.append(it->Name);
		parseContext_->RegisterEvent(asmm::ErrorMessage(0x300501, s, _T("Input parse error")));
	}
}