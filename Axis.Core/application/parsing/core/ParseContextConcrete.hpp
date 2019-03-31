#pragma once
#include "application/parsing/core/ParseContext.hpp"
#include "AxisString.hpp"
#include "services/messaging/EventMessage.hpp"
#include "services/messaging/CollectorEndpoint.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class ParseContextConcrete : public ParseContext
{
public:
  ParseContextConcrete(void);
  ~ParseContextConcrete(void);

  virtual RunMode GetRunMode(void) const;
  virtual axis::String GetParseSourceName(void) const;
  virtual unsigned long GetParseSourceCursorLocation(void) const;
  virtual int GetCurrentRoundIndex(void) const;
  virtual axis::application::jobs::AnalysisStep *GetStepOnFocus(void);
  virtual const axis::application::jobs::AnalysisStep *GetStepOnFocus(void) const;
  void SetStepOnFocus(axis::application::jobs::AnalysisStep *step);
  virtual int GetStepOnFocusIndex(void) const;
  virtual void SetStepOnFocusIndex(int index);

  void SetRunMode(RunMode mode);
  void AdvanceRound(void);
  void SetParseSourceName(const axis::String& sourceName);
  void SetParseSourceCursorLocation(unsigned long lineIndex);

  void ClearEventStatistics(void); // moved to public
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} }	} } // namespace axis::application::parsing::core
