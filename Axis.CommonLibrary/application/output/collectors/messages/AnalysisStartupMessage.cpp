#include "AnalysisStartupMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace asmm = axis::services::messaging;

const asmm::Message::id_type aaocm::AnalysisStartupMessage::BaseId = 200;

aaocm::AnalysisStartupMessage::AnalysisStartupMessage( void) :
ResultMessage(BaseId)
{
	// nothing to do here
}

aaocm::AnalysisStartupMessage::AnalysisStartupMessage( const axis::String& description ) :
ResultMessage(BaseId, description)
{
	// nothing to do here
}

aaocm::AnalysisStartupMessage::~AnalysisStartupMessage( void )
{
	// nothing to do here
}

bool aaocm::AnalysisStartupMessage::IsOfKind( const ResultMessage& message )
{
	return message.GetId() == BaseId;
}

void aaocm::AnalysisStartupMessage::DoDestroy( void ) const
{
	delete this;
}

asmm::Message& aaocm::AnalysisStartupMessage::DoClone( id_type id ) const
{
	return *new AnalysisStartupMessage(GetDescription());
}

