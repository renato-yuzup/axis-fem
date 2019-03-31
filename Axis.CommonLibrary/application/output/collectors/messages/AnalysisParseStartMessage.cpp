#include "AnalysisParseStartMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace asmm = axis::services::messaging;

const asmm::Message::id_type aaocm::AnalysisParseStartMessage::BaseId = 50;

aaocm::AnalysisParseStartMessage::AnalysisParseStartMessage( void ) :
ResultMessage(BaseId)
{
	// nothing to do here
}

aaocm::AnalysisParseStartMessage::AnalysisParseStartMessage( const axis::String& description ) :
ResultMessage(BaseId, description)
{
	// nothing to do here
}

aaocm::AnalysisParseStartMessage::~AnalysisParseStartMessage( void )
{
	// nothing to do here
}

bool aaocm::AnalysisParseStartMessage::IsOfKind( const ResultMessage& message )
{
	return message.GetId() == BaseId;
}

void aaocm::AnalysisParseStartMessage::DoDestroy( void ) const
{
	// nothing to do here
}

asmm::Message& aaocm::AnalysisParseStartMessage::DoClone( id_type ) const
{
	return *new AnalysisParseStartMessage(GetDescription());
}