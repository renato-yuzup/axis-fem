#include "ParseFinishMessage.hpp"

namespace aaam = axis::application::agents::messages;

const axis::services::messaging::Message::id_type aaam::ParseFinishMessage::BaseId = 9001;

aaam::ParseFinishMessage::ParseFinishMessage( const axis::application::jobs::StructuralAnalysis& analysis, 
                                              long errorCount, 
                                              long warningCount, 
                                              long readRoundCount ) :
  ResultMessage(BaseId), analysis_(&analysis), errorCount_(errorCount), 
  warningCount_(warningCount), readRoundCount_(readRoundCount)
{
  // nothing to do here
}

aaam::ParseFinishMessage::~ParseFinishMessage( void )
{
  // nothing to do here
}

bool aaam::ParseFinishMessage::IsOfKind( const ResultMessage& message )
{
  return message.GetId() == BaseId;
}

long axis::application::agents::messages::ParseFinishMessage::GetErrorCount( void ) const
{
  return errorCount_;
}

long axis::application::agents::messages::ParseFinishMessage::GetWarningCount( void ) const
{
  return warningCount_;
}

long axis::application::agents::messages::ParseFinishMessage::GetReadRoundCount( void ) const
{
  return readRoundCount_;
}

const axis::application::jobs::StructuralAnalysis& axis::application::agents::messages::ParseFinishMessage::GetAnalysis( void ) const
{
  return *analysis_;
}

void aaam::ParseFinishMessage::DoDestroy( void ) const
{
  delete this;
}

axis::services::messaging::Message& aaam::ParseFinishMessage::DoClone( id_type id ) const
{
  return *new ParseFinishMessage(*analysis_, errorCount_, warningCount_, readRoundCount_);
}
