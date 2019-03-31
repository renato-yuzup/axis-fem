#if defined DEBUG || defined _DEBUG
#pragma once
#include "application/parsing/core/ParseContext.hpp"
#include "application/jobs/AnalysisStep.hpp"

namespace axis { namespace unit_tests { namespace orange {

class MockParseContext : public axis::application::parsing::core::ParseContext
{
public:
  MockParseContext(void);
  virtual ~MockParseContext(void);

  virtual RunMode GetRunMode( void ) const;

  virtual axis::String GetParseSourceName( void ) const;

  virtual unsigned long GetParseSourceCursorLocation( void ) const;

  virtual int GetCurrentRoundIndex( void ) const;

  virtual axis::application::jobs::AnalysisStep * GetStepOnFocus( void );

  virtual const axis::application::jobs::AnalysisStep * GetStepOnFocus( void ) const;

  virtual void SetStepOnFocus( axis::application::jobs::AnalysisStep *step );

  virtual int GetStepOnFocusIndex( void ) const;

  virtual void SetStepOnFocusIndex( int index );
private:
  axis::application::jobs::AnalysisStep *currentStep_;
  int stepIndex_;
};

} } }

#endif