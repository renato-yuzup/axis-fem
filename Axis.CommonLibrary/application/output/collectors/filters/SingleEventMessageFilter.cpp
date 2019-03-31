#include "SingleEventMessageFilter.hpp"
#include "application/output/collectors/messages/AnalysisParseStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisParseFinishMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaocf = axis::application::output::collectors::filters;
namespace asmm = axis::services::messaging;

aaocf::SingleEventMessageFilter::SingleEventMessageFilter( asmm::Message::id_type eventMessageId )
{
	_messageId = eventMessageId;
}

aaocf::SingleEventMessageFilter::~SingleEventMessageFilter( void )
{
	// nothing to do here
}

bool aaocf::SingleEventMessageFilter::IsEventMessageFiltered( const asmm::EventMessage& message )
{
	// accept only the expected message
	return message.GetId() != _messageId;
}

bool aaocf::SingleEventMessageFilter::IsResultMessageFiltered( const asmm::ResultMessage& )
{
	// reject any result message
	return true;
}

void aaocf::SingleEventMessageFilter::Destroy( void ) const
{
	delete this;
}

asmm::filters::MessageFilter& aaocf::SingleEventMessageFilter::Clone( void ) const
{
	return *new SingleEventMessageFilter(_messageId);
}

