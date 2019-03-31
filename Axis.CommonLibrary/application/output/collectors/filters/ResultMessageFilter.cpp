#include "ResultMessageFilter.hpp"
#include "application/output/collectors/messages/AnalysisParseFinishMessage.hpp"
#include "application/output/collectors/messages/AnalysisParseStartMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaocf = axis::application::output::collectors::filters;
namespace asmm = axis::services::messaging;

aaocf::ResultMessageFilter::ResultMessageFilter( void )
{
	// nothing to do here
}

aaocf::ResultMessageFilter::~ResultMessageFilter( void )
{
	// nothing to do here
}

bool aaocf::ResultMessageFilter::IsEventMessageFiltered( const asmm::EventMessage& )
{
	// reject any event message
	return true;
}

bool aaocf::ResultMessageFilter::IsResultMessageFiltered( const asmm::ResultMessage& )
{
	// accept result messages
	return false;
}

void aaocf::ResultMessageFilter::Destroy( void ) const
{
	delete this;
}

asmm::filters::MessageFilter& aaocf::ResultMessageFilter::Clone( void ) const
{
	return *new ResultMessageFilter();
}

