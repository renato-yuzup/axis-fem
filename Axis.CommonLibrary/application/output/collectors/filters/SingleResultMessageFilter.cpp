#include "SingleResultMessageFilter.hpp"
#include "application/output/collectors/messages/AnalysisParseStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisParseFinishMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaocf = axis::application::output::collectors::filters;
namespace asmm = axis::services::messaging;

aaocf::SingleResultMessageFilter::SingleResultMessageFilter( asmm::Message::id_type resultMessageId )
{
	_messageId = resultMessageId;
}

aaocf::SingleResultMessageFilter::~SingleResultMessageFilter( void )
{
	// nothing to do here
}

bool aaocf::SingleResultMessageFilter::IsEventMessageFiltered( const asmm::EventMessage& )
{
	// reject any event message
	return true;
}

bool aaocf::SingleResultMessageFilter::IsResultMessageFiltered( const asmm::ResultMessage& message )
{
	// accept only the expected message
	return message.GetId() != _messageId;
}

void aaocf::SingleResultMessageFilter::Destroy( void ) const
{
	delete this;
}

asmm::filters::MessageFilter& aaocf::SingleResultMessageFilter::Clone( void ) const
{
	return *new SingleResultMessageFilter(_messageId);
}

