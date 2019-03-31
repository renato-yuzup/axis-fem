#include "ParseRoundEndMessage.hpp"

namespace aaam = axis::application::agents::messages;
namespace aaj  = axis::application::jobs;

const axis::services::messaging::Message::id_type aaam::ParseRoundEndMessage::BaseId = 9002;


aaam::ParseRoundEndMessage::ParseRoundEndMessage( const aaj::StructuralAnalysis& analysis, 
                                                  long errorCount, long warningCount, 
                                                  long currentRount, long definedSymbolCount, 
                                                  long undefinedSymbolCount ) :
  ResultMessage(BaseId), analysis_(&analysis), errorCount_(errorCount), 
  warningCount_(warningCount), roundIndex_(currentRount),
  definedSymbolCount_(definedSymbolCount), undefinedSymbolCount_(undefinedSymbolCount)
{
  // nothing to do here
}

aaam::ParseRoundEndMessage::~ParseRoundEndMessage( void )
{
  // nothing to do here
}

bool aaam::ParseRoundEndMessage::IsOfKind( const ResultMessage& message )
{
  return message.GetId() == BaseId;
}

long aaam::ParseRoundEndMessage::GetErrorCount( void ) const
{
  return errorCount_;
}

long aaam::ParseRoundEndMessage::GetWarningCount( void ) const
{
  return warningCount_;
}

long aaam::ParseRoundEndMessage::GetRoundIndex( void ) const
{
  return roundIndex_;
}

long aaam::ParseRoundEndMessage::GetUndefinedSymbolCount( void ) const
{
  return undefinedSymbolCount_;
}

long aaam::ParseRoundEndMessage::GetDefinedSymbolCount( void ) const
{
  return definedSymbolCount_;
}

const aaj::StructuralAnalysis& aaam::ParseRoundEndMessage::GetAnalysis( void ) const
{
  return *analysis_;
}

void aaam::ParseRoundEndMessage::DoDestroy( void ) const
{
  delete this;
}

axis::services::messaging::Message& aaam::ParseRoundEndMessage::DoClone( id_type id ) const
{
  return *new ParseRoundEndMessage(*analysis_, errorCount_, warningCount_, 
                                   roundIndex_, definedSymbolCount_, undefinedSymbolCount_);
}

