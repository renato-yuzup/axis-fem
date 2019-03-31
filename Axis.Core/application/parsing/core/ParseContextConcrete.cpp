#include "ParseContextConcrete.hpp"
#include "ParseContextConcrete_Pimpl.hpp"

namespace aaj = axis::application::jobs;
namespace aapc = axis::application::parsing::core;

aapc::ParseContextConcrete::ParseContextConcrete(void)
{
  pimpl_ = new Pimpl();
  pimpl_->runMode = ParseContext::kTrialMode;
  pimpl_->stepOnFocus = NULL;
  pimpl_->parseSourceCursorLocation = 0;
  pimpl_->roundIndex = 0;
  pimpl_->stepOnFocusIndex = -1;
}

aapc::ParseContextConcrete::~ParseContextConcrete(void)
{
  delete pimpl_;
}

aapc::ParseContext::RunMode aapc::ParseContextConcrete::GetRunMode( void ) const
{
  return pimpl_->runMode;
}

axis::String aapc::ParseContextConcrete::GetParseSourceName( void ) const
{
  return pimpl_->parseSourceName;
}

unsigned long aapc::ParseContextConcrete::GetParseSourceCursorLocation( void ) const
{
  return pimpl_->parseSourceCursorLocation;
}

int aapc::ParseContextConcrete::GetCurrentRoundIndex( void ) const
{
  return pimpl_->roundIndex;
}

aaj::AnalysisStep *aapc::ParseContextConcrete::GetStepOnFocus( void )
{
  return pimpl_->stepOnFocus;
}

const aaj::AnalysisStep *aapc::ParseContextConcrete::GetStepOnFocus( void ) const
{
  return pimpl_->stepOnFocus;
}

void aapc::ParseContextConcrete::SetStepOnFocus( aaj::AnalysisStep *step )
{
  pimpl_->stepOnFocus = step;
}

int aapc::ParseContextConcrete::GetStepOnFocusIndex( void ) const
{
  return pimpl_->stepOnFocusIndex;
}

void aapc::ParseContextConcrete::SetStepOnFocusIndex( int index )
{
  pimpl_->stepOnFocusIndex = index;
}

void aapc::ParseContextConcrete::SetRunMode( RunMode mode )
{
  pimpl_->runMode = mode;
}

void aapc::ParseContextConcrete::AdvanceRound( void )
{
  ParseContext::AdvanceRound();
  pimpl_->roundIndex++;
  pimpl_->stepOnFocusIndex = -1;
}

void aapc::ParseContextConcrete::SetParseSourceName( const axis::String& sourceName )
{
  pimpl_->parseSourceName = sourceName;
}

void aapc::ParseContextConcrete::SetParseSourceCursorLocation( unsigned long lineIndex )
{
  pimpl_->parseSourceCursorLocation = lineIndex;
}

void aapc::ParseContextConcrete::ClearEventStatistics( void )
{
  ParseContext::ClearEventStatistics();
}
