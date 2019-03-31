#include "AnalysisParseFinishMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace asmm = axis::services::messaging;

const asmm::Message::id_type aaocm::AnalysisParseFinishMessage::BaseId = 51;

aaocm::AnalysisParseFinishMessage::AnalysisParseFinishMessage( void ) :
ResultMessage(BaseId)
{
	// nothing to do here
}

aaocm::AnalysisParseFinishMessage::AnalysisParseFinishMessage( const axis::String& description ) :
ResultMessage(BaseId, description)
{
	// nothing to do here
}

aaocm::AnalysisParseFinishMessage::~AnalysisParseFinishMessage( void )
{
	// nothing to do here
}

bool aaocm::AnalysisParseFinishMessage::IsOfKind( const ResultMessage& message )
{
	return message.GetId() == BaseId;
}

void aaocm::AnalysisParseFinishMessage::DoDestroy( void ) const
{
	// nothing to do here
}

asmm::Message& aaocm::AnalysisParseFinishMessage::DoClone( id_type ) const
{
	return *new AnalysisParseFinishMessage(GetDescription());
}