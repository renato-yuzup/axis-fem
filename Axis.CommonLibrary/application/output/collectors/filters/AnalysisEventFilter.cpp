#include "AnalysisEventFilter.hpp"
#include "application/output/collectors/messages/AnalysisParseStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisParseFinishMessage.hpp"
#include "application/output/collectors/messages/AnalysisEndMessage.hpp"
#include "application/output/collectors/messages/AnalysisStartupMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaocf = axis::application::output::collectors::filters;
namespace asmm = axis::services::messaging;

aaocf::AnalysisEventFilter::AnalysisEventFilter( void )
{
	_isFiltering = true;
}

aaocf::AnalysisEventFilter::~AnalysisEventFilter( void )
{
	// nothing to do here
}

bool aaocf::AnalysisEventFilter::IsEventMessageFiltered( const asmm::EventMessage& message )
{
  if (message.GetTraceInformation().Contains(1000, _T("Solver")))
  {
    return false;
  }
  else if (message.GetTraceInformation().Contains(500, _T("Parser")))
  {
    return false;
  }
	// we only accept events that are fired during an analysis
	return _isFiltering || EventLogMessageFilter::IsEventMessageFiltered(message);
}

bool aaocf::AnalysisEventFilter::IsResultMessageFiltered( const asmm::ResultMessage& message )
{
	// only four messages are of our interest: the one which
	// indicates the job start and another fired on its end;
	// and the ones which indicates the start and end of parsing
	if (aaocm::AnalysisParseStartMessage::IsOfKind(message))
	{	// job has started
		_isFiltering = false;
	}
	else if (aaocm::AnalysisParseFinishMessage::IsOfKind(message))
	{	// job has finished, stop accepting events
		_isFiltering = true;
	}
	if (aaocm::AnalysisStartupMessage::IsOfKind(message))
	{	// job has started
		_isFiltering = false;
	}
	else if (aaocm::AnalysisEndMessage::IsOfKind(message))
	{	// job has finished, stop accepting events
		_isFiltering = true;
	}
  else if (message.GetTraceInformation().Contains(1000, _T("Solver")))
  {
    return false;
  }
  else if (message.GetTraceInformation().Contains(500, _T("Parser")))
  {
    return false;
  }

	// don't allow any result messages to pass
	return true;
}

void aaocf::AnalysisEventFilter::Destroy( void ) const
{
	delete this;
}

asmm::filters::MessageFilter& aaocf::AnalysisEventFilter::Clone( void ) const
{
	return *new AnalysisEventFilter(*this);
}

