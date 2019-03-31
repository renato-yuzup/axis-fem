#pragma once
#include "ParserAgent.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "messages/ParseFinishMessage.hpp"
#include "messages/ParseRoundEndMessage.hpp"
#include "services/messaging/MessageListener.hpp"
#include "application/jobs/StructuralAnalysis.hpp"

namespace axis { namespace application { namespace agents {

class ParserAgent::ParserAgentDispatcher : public axis::services::messaging::MessageListener
{
public:
  ParserAgentDispatcher(ParserAgent *parent);
  ~ParserAgentDispatcher(void);

  void LogParseStart(const axis::application::jobs::StructuralAnalysis& analysis, 
                     const axis::String& masterInputFile, 
                     const axis::String& baseIncludePath);
protected:
  virtual void DoProcessResultMessage(axis::services::messaging::ResultMessage& volatileMessage);
private:
  void LogParseEndSummary(const axis::application::agents::messages::ParseFinishMessage& message);
  void LogParsePartialSummary(const axis::application::agents::messages::ParseRoundEndMessage& message);
  void PrintModelSummary(const axis::application::jobs::StructuralAnalysis& analysis);
  void PrintOutputList(const axis::application::jobs::StructuralAnalysis& analysis);
  ParserAgent *parent_;
};

} } } // namespace axis::application::agents
