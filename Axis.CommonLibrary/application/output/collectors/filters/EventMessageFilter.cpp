#include "EventMessageFilter.hpp"
#include "application/output/collectors/messages/AnalysisParseStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisParseFinishMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaocf = axis::application::output::collectors::filters;
namespace asmm = axis::services::messaging;

aaocf::EventMessageFilter::EventMessageFilter( void )
{
	// nothing to do here
}

aaocf::EventMessageFilter::~EventMessageFilter( void )
{
	// nothing to do here
}

bool aaocf::EventMessageFilter::IsEventMessageFiltered( const asmm::EventMessage& )
{
	// reject any event message
	return true;
}

bool aaocf::EventMessageFilter::IsResultMessageFiltered( const asmm::ResultMessage& )
{
	// accept result messages
	return false;
}

void aaocf::EventMessageFilter::Destroy( void ) const
{
	delete this;
}

asmm::filters::MessageFilter& aaocf::EventMessageFilter::Clone( void ) const
{
	return *new EventMessageFilter();
}

