#include "ModalAnalysisInfo.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfRangeException.hpp"

namespace ada = axis::domain::analyses;

ada::ModalAnalysisInfo::ModalAnalysisInfo(long firstModeIndex, long lastModeIndex)
{
  if (firstModeIndex > lastModeIndex)
  {
    throw axis::foundation::ArgumentException(_T("Invalid mode range."));
  }
  firstMode_ = firstModeIndex;
  lastMode_ = lastModeIndex;
}

ada::ModalAnalysisInfo::~ModalAnalysisInfo(void)
{
  // nothing to do here
}

void axis::domain::analyses::ModalAnalysisInfo::Destroy( void ) const
{
  delete this;
}

ada::AnalysisInfo::AnalysisType ada::ModalAnalysisInfo::GetAnalysisType( void ) const
{
  return ModalAnalysis;
}

real axis::domain::analyses::ModalAnalysisInfo::GetCurrentFrequency( void ) const
{
  return currentFrequency_;
}

void axis::domain::analyses::ModalAnalysisInfo::SetCurrentFrequency( real freq )
{
  currentFrequency_ = freq;
}

long axis::domain::analyses::ModalAnalysisInfo::GetCurrentModeIndex( void ) const
{
  return currentMode_;
}

void axis::domain::analyses::ModalAnalysisInfo::SetCurrentModeIndex( long index )
{
  if (currentMode_ < firstMode_ || currentMode_ > lastMode_)
  {
    throw axis::foundation::OutOfRangeException(_T("Mode out of range."));
  }
  currentMode_ = index;
}

long axis::domain::analyses::ModalAnalysisInfo::GetFirstModeIndex( void ) const
{
  return firstMode_;
}

long axis::domain::analyses::ModalAnalysisInfo::GetLastModeIndex( void ) const
{
  return lastMode_;
}

ada::AnalysisInfo& ada::ModalAnalysisInfo::Clone( void ) const
{
  ModalAnalysisInfo& info = *new ModalAnalysisInfo(firstMode_, lastMode_);
  info.SetCurrentModeIndex(currentMode_);
  info.SetCurrentFrequency(currentFrequency_);
  return info;
}
