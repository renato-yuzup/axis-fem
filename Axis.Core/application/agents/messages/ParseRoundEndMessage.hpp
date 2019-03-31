#pragma once
#include "services/messaging/ResultMessage.hpp"
#include "application/jobs/StructuralAnalysis.hpp"

namespace axis { namespace application { namespace agents { namespace messages {

class ParseRoundEndMessage : public axis::services::messaging::ResultMessage
{
public:
  ParseRoundEndMessage(const axis::application::jobs::StructuralAnalysis& analysis, 
                        long errorCount, long warningCount, long currentRount, 
                        long definedSymbolCount, long undefinedSymbolCount);
  ~ParseRoundEndMessage(void);
  static bool IsOfKind(const ResultMessage& message);
  long GetErrorCount(void) const;
  long GetWarningCount(void) const;
  long GetRoundIndex(void) const;
  long GetUndefinedSymbolCount(void) const;
  long GetDefinedSymbolCount(void) const;
  const axis::application::jobs::StructuralAnalysis& GetAnalysis(void) const;

  static const Message::id_type BaseId;
private:
  virtual void DoDestroy( void ) const;
  virtual Message& DoClone( id_type id ) const;

  long errorCount_, warningCount_, roundIndex_;
  long definedSymbolCount_, undefinedSymbolCount_;
  const axis::application::jobs::StructuralAnalysis *analysis_;
};

} } } } // namespace axis::application::agents::messages
