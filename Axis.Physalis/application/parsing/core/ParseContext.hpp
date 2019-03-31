#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/fwd/jobs_fwd.hpp"
#include "application/fwd/parsing_fwd.hpp"
#include "AxisString.hpp"
#include "services/messaging/EventMessage.hpp"
#include "services/messaging/CollectorEndpoint.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class AXISPHYSALIS_API ParseContext : public axis::services::messaging::CollectorEndpoint
{
public:
	enum RunMode
	{
		kTrialMode = 0,
		kCollateMode = 2,
		kInspectionMode = 4
	};
  class EventStatistic;

  ParseContext(void);
	~ParseContext(void);
	virtual RunMode GetRunMode(void) const = 0;
  virtual axis::String GetParseSourceName(void) const = 0;
  virtual unsigned long GetParseSourceCursorLocation(void) const = 0;
  virtual int GetCurrentRoundIndex(void) const = 0;
  virtual axis::application::jobs::AnalysisStep *GetStepOnFocus(void) = 0;
  virtual const axis::application::jobs::AnalysisStep *GetStepOnFocus(void) const = 0;
  virtual void SetStepOnFocus(axis::application::jobs::AnalysisStep *step) = 0;
  virtual int GetStepOnFocusIndex(void) const = 0;
  virtual void SetStepOnFocusIndex(int index) = 0;
  void RegisterEvent(axis::services::messaging::EventMessage& event) const;
	const EventStatistic& EventSummary(void) const;
  SymbolTable& Symbols(void);
  Sketchbook& Sketches(void);
  EntityLabeler& Labels(void);
  const SymbolTable& Symbols(void) const;
  const Sketchbook& Sketches(void) const;
  const EntityLabeler& Labels(void) const;
	static const int MaxAllowableErrorCount;
	static const int SourceId;
protected:
  void ClearEventStatistics(void);
  void AdvanceRound(void);
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} }	} } // namespace axis::application::parsing::core
