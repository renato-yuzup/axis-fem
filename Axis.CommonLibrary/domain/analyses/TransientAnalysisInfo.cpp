#include "TransientAnalysisInfo.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfRangeException.hpp"

namespace ada = axis::domain::analyses;

ada::TransientAnalysisInfo::TransientAnalysisInfo( real startTime, real endTime )
{
  if (startTime > endTime)
  {
    throw axis::foundation::ArgumentException(_T("Invalid analysis start/end time."));
  }
  startTime_ = startTime; endTime_ = endTime;
  iterationIndex_ = 0;
}

ada::TransientAnalysisInfo::~TransientAnalysisInfo( void )
{
  // nothing to do here
}

void axis::domain::analyses::TransientAnalysisInfo::Destroy( void ) const
{
  delete this;
}

ada::AnalysisInfo& ada::TransientAnalysisInfo::Clone( void ) const
{
  TransientAnalysisInfo& info = *new TransientAnalysisInfo(startTime_, endTime_);
  info.SetLastTimeStep(lastTimeStep_);
  info.SetCurrentAnalysisTime(currentTime_);
  info.SetIterationIndex(iterationIndex_);
  return info;
}

ada::AnalysisInfo::AnalysisType ada::TransientAnalysisInfo::GetAnalysisType( void ) const
{
  return TransientAnalysis;
}

real ada::TransientAnalysisInfo::GetStartTime( void ) const
{
  return startTime_;
}

real ada::TransientAnalysisInfo::GetEndTime( void ) const
{
  return endTime_;
}

real ada::TransientAnalysisInfo::GetCurrentAnalysisTime( void ) const
{
  return currentTime_;
}

void ada::TransientAnalysisInfo::SetCurrentAnalysisTime( real newTime )
{
  if (newTime < startTime_)
  {
    throw axis::foundation::OutOfRangeException(_T("Invalid time."));
  }
  currentTime_ = newTime;
}

real ada::TransientAnalysisInfo::GetLastTimeStep( void ) const
{
  return lastTimeStep_;
}

void ada::TransientAnalysisInfo::SetLastTimeStep( real newTimeStep )
{
  lastTimeStep_ = newTimeStep;
}

uint64 ada::TransientAnalysisInfo::GetIterationIndex(void) const
{
  return iterationIndex_;
}

void ada::TransientAnalysisInfo::SetIterationIndex(uint64 index)
{
  iterationIndex_ = index;
}
