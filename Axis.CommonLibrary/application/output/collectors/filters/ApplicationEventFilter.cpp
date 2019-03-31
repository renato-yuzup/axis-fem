#include "ApplicationEventFilter.hpp"
#include "application/output/collectors/messages/AnalysisParseStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisParseFinishMessage.hpp"
#include "application/output/collectors/messages/AnalysisStartupMessage.hpp"
#include "application/output/collectors/messages/AnalysisEndMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaocf = axis::application::output::collectors::filters;
namespace asmm = axis::services::messaging;

aaocf::ApplicationEventFilter::ApplicationEventFilter( void )
{
	_isFiltering = false;
}

aaocf::ApplicationEventFilter::~ApplicationEventFilter( void )
{
	// nothing to do here
}

bool aaocf::ApplicationEventFilter::IsEventMessageFiltered( const asmm::EventMessage& message)
{
  if (message.GetTraceInformation().Contains(1000, _T("Solver")))
  {
    return true;
  }
  else if (message.GetTraceInformation().Contains(500, _T("Parser")))
  {
    return true;
  }
	// we only accept events that are NOT fired during an analysis
	return _isFiltering || EventLogMessageFilter::IsEventMessageFiltered(message);
}

bool aaocf::ApplicationEventFilter::IsResultMessageFiltered( const asmm::ResultMessage& message )
{
	// only two messages are of our interest: the one which
	// indicates the job start and another fired on its end
	if (aaocm::AnalysisParseStartMessage::IsOfKind(message))
	{	// job has started
		_isFiltering = true;
	}
	else if (aaocm::AnalysisParseFinishMessage::IsOfKind(message))
	{	// job has finished, start accepting events again
		_isFiltering = false;
	}
	if (aaocm::AnalysisStartupMessage::IsOfKind(message))
	{	// job has started
		_isFiltering = true;
	}
	else if (aaocm::AnalysisEndMessage::IsOfKind(message))
	{	// job has finished, stop accepting events
		_isFiltering = false;
	}
  else if (message.GetTraceInformation().Contains(1000, _T("Solver")))
  {
    return true;
  }
  else if (message.GetTraceInformation().Contains(500, _T("Parser")))
  {
    return true;
  }

	// don't allow any result messages to pass
	return true;
}

void aaocf::ApplicationEventFilter::Destroy( void ) const
{
	delete this;
}

asmm::filters::MessageFilter& aaocf::ApplicationEventFilter::Clone( void ) const
{
	return *new ApplicationEventFilter(*this);
}

