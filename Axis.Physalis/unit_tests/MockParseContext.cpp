#if defined DEBUG || defined _DEBUG 
#include "MockParseContext.hpp"

namespace aapc = axis::application::parsing::core;
namespace aup = axis::unit_tests::physalis;

aup::MockParseContext::MockParseContext( void )
{
  currentStep_ = NULL;
  stepIndex_ = 0;
}

aup::MockParseContext::~MockParseContext( void )
{
  // nothing to do here
}

aapc::ParseContext::RunMode aup::MockParseContext::GetRunMode( void ) const
{
  return aapc::ParseContext::kTrialMode;
}
axis::String aup::MockParseContext::GetParseSourceName( void ) const
{
  return _T("");
}

unsigned long aup::MockParseContext::GetParseSourceCursorLocation( void ) const
{
  return 0;
}

int aup::MockParseContext::GetCurrentRoundIndex( void ) const
{
  return 0;
}

axis::application::jobs::AnalysisStep * aup::MockParseContext::GetStepOnFocus( void )
{
  return currentStep_;
}
const axis::application::jobs::AnalysisStep * aup::MockParseContext::GetStepOnFocus( void ) const
{
  return currentStep_;
}

void aup::MockParseContext::SetStepOnFocus( axis::application::jobs::AnalysisStep *step )
{
  currentStep_ = step;
}

int aup::MockParseContext::GetStepOnFocusIndex( void ) const
{
  return stepIndex_;
}

void aup::MockParseContext::SetStepOnFocusIndex( int index )
{
  stepIndex_ = index;
}

#endif