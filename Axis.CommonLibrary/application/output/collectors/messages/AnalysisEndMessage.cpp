#include "AnalysisEndMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace asmm = axis::services::messaging;

const asmm::Message::id_type aaocm::AnalysisEndMessage::BaseId = 250;

aaocm::AnalysisEndMessage::AnalysisEndMessage( void ) :
ResultMessage(BaseId)
{
	// nothing to do here
}

aaocm::AnalysisEndMessage::AnalysisEndMessage( const axis::String& description ) :
ResultMessage(BaseId, description)
{
	// nothing to do here
}

aaocm::AnalysisEndMessage::~AnalysisEndMessage( void )
{
	// nothing to do here
}

bool aaocm::AnalysisEndMessage::IsOfKind( const ResultMessage& message )
{
	return message.GetId() == BaseId;
}

void aaocm::AnalysisEndMessage::DoDestroy( void ) const
{
	// nothing to do here
}

asmm::Message& aaocm::AnalysisEndMessage::DoClone( id_type ) const
{
	return *new AnalysisEndMessage(GetDescription());
}
