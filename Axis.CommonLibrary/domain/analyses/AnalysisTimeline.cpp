#include "AnalysisTimeline.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "AnalysisTimeline_Pimpl.hpp"

namespace ada = axis::domain::analyses;
namespace af = axis::foundation;

static const real time_resolution = 1e-12;
static const real time_tolerance = 1e-9;

ada::AnalysisTimeline::AnalysisTimeline( real startTime, real endTime )
{
	Init();
  startTime_ = startTime;
  endTime_ = endTime;
}

ada::AnalysisTimeline::AnalysisTimeline( real startTime, real endTime, 
                                         real currentTime, real lastTimeIncrement )
{
	Init();
  startTime_ = startTime;
  endTime_ = endTime;
  currentTime_ = currentTime;
  lastTimestep_ = lastTimeIncrement;
}

ada::AnalysisTimeline::AnalysisTimeline( const AnalysisTimeline& other )
{
	Copy(other);
}

ada::AnalysisTimeline::~AnalysisTimeline( void )
{
	delete pimpl_;
}

void ada::AnalysisTimeline::Destroy( void ) const
{
	delete this;
}

ada::AnalysisTimeline& ada::AnalysisTimeline::Create( real startTime, real endTime )
{
	return *new ada::AnalysisTimeline(startTime, endTime);
}

ada::AnalysisTimeline& ada::AnalysisTimeline::Create( real startTime, real endTime, 
                                                      real currentTime, real lastTimeIncrement )
{
	return *new ada::AnalysisTimeline(startTime, endTime, currentTime, lastTimeIncrement);
}

void ada::AnalysisTimeline::Init( void )
{
  pimpl_ = new Pimpl();
	currentTime_ = 0;
	iterationIndex_ = 0;
	startTime_ = 0;
	endTime_ = 0;
	lastTimestep_ = 0;
	nextTimestep_ = 0;

	pimpl_->hasCrossedSnapshotMark = false;
	pimpl_->currentSnapshotMarkTime = 0;
	pimpl_->nextSnapshotIndex = 0;
}

void ada::AnalysisTimeline::Copy( const AnalysisTimeline& other )
{
	currentTime_ = other.GetCurrentSnapshotTime();
	iterationIndex_ = other.IterationIndex();
	startTime_ = other.StartTime();
	endTime_ = other.EndTime();
	lastTimestep_ = other.LastTimeIncrement();
	nextTimestep_ = other.NextTimeIncrement();

	pimpl_->nextSnapshotIndex = other.pimpl_->nextSnapshotIndex;
	pimpl_->currentSnapshotMarkTime = other.pimpl_->currentSnapshotMarkTime;
	pimpl_->hasCrossedSnapshotMark = other.pimpl_->hasCrossedSnapshotMark;

	// copy snapshot marks
	ClearSnapshotMarks();
  pimpl_->snapshotMarks.insert(pimpl_->snapshotMarks.begin(),
                               other.pimpl_->snapshotMarks.begin(), 
                               other.pimpl_->snapshotMarks.end());
}

ada::AnalysisTimeline::iteration_index ada::AnalysisTimeline::IterationIndex( void ) const
{
	return iterationIndex_;
}

ada::AnalysisTimeline::iteration_index& ada::AnalysisTimeline::IterationIndex( void )
{
	return iterationIndex_;
}

real ada::AnalysisTimeline::StartTime( void ) const
{
	return startTime_;
}

real ada::AnalysisTimeline::EndTime( void ) const
{
	return endTime_;
}

real ada::AnalysisTimeline::NextTimeIncrement( void ) const
{
	return nextTimestep_;
}

real& ada::AnalysisTimeline::NextTimeIncrement( void )
{
	return nextTimestep_;
}

real ada::AnalysisTimeline::LastTimeIncrement( void ) const
{
	return lastTimestep_;
}

real ada::AnalysisTimeline::GetCurrentTimeMark( void ) const
{
	return currentTime_;
}

void ada::AnalysisTimeline::Tick( void )
{
	// we cannot advance further the end of the timeline
	if (currentTime_ > endTime_)
	{
		throw af::InvalidOperationException(_T("Cannot advance past the end of the timeline."));
	}

	iterationIndex_++;
	currentTime_ += nextTimestep_;
	lastTimestep_ = nextTimestep_;
	nextTimestep_ = 0;

	// if we were over a snapshot mark, move away
	if (pimpl_->hasCrossedSnapshotMark)
	{
		pimpl_->hasCrossedSnapshotMark = false;
		pimpl_->nextSnapshotIndex++;
		pimpl_->currentSnapshotMarkTime = 0;
	}

	// loop over snapshot marks in the case we have advanced too
	// much in time

	// check if we crossed a snapshot mark
	bool overMark = true;
	size_type i;
	for (i = pimpl_->nextSnapshotIndex; i < pimpl_->snapshotMarks.size() && overMark; ++i)
	{
		real markTime = pimpl_->snapshotMarks[i]->GetTime();
		// we state that a mark is crossed in two situations: 
		// 1) when the current time mark is almost over the snapshot
		// mark within a certain tolerance, and
		// 2) when we effectively crossed the snapshot mark
		overMark = (abs(currentTime_ - markTime) / lastTimestep_ <= time_tolerance);
		overMark |= (currentTime_ - markTime >= 0);

		// if we have crossed a mark, update internal variables
		if (overMark)
		{
			pimpl_->currentSnapshotMarkTime = markTime;
			pimpl_->hasCrossedSnapshotMark = true;
			pimpl_->nextSnapshotIndex = i;
		}
	}
}

ada::AnalysisTimeline& ada::AnalysisTimeline::operator=( const AnalysisTimeline& other )
{
	Copy(other);
	return *this;
}

void ada::AnalysisTimeline::AddSnapshotMark( const SnapshotMark& mark )
{
	// check if this snapshot mark is after the last one we have
	bool ok = true;
	if (pimpl_->snapshotMarks.size() > 0)
	{
		real lastTime = pimpl_->snapshotMarks[pimpl_->snapshotMarks.size() - 1]->GetTime();
		ok = (mark.GetTime() - lastTime) > time_resolution;
	}
	if (!ok)
	{
		throw af::ArgumentException(_T("mark"));
	}
	pimpl_->snapshotMarks.push_back(&mark.Clone());
}

void ada::AnalysisTimeline::ClearSnapshotMarks( void )
{
	for (size_type i = 0; i < pimpl_->snapshotMarks.size(); ++i)
	{
		pimpl_->snapshotMarks[i]->Destroy();
	}
	pimpl_->snapshotMarks.clear();
}

size_type ada::AnalysisTimeline::SnapshotMarkCount( void ) const
{
	return (size_type)pimpl_->snapshotMarks.size();
}

real ada::AnalysisTimeline::GetCurrentSnapshotTime( void ) const
{
	if (!HasCrossedSnapshotMark())
	{
		throw af::InvalidOperationException(_T("No snapshot marks has been crossed since last iteration."));
	}
	return pimpl_->currentSnapshotMarkTime;
}

bool ada::AnalysisTimeline::HasCrossedSnapshotMark( void ) const
{
	return pimpl_->hasCrossedSnapshotMark;
}

void ada::AnalysisTimeline::Reset( void )
{
	currentTime_ = startTime_;
	iterationIndex_ = 0;
	lastTimestep_ = 0;
	nextTimestep_ = 0;
	pimpl_->hasCrossedSnapshotMark = false;
	pimpl_->nextSnapshotIndex = 0;
	pimpl_->currentSnapshotMarkTime = 0;

	// loop over snapshot marks in the case we have advanced too
	// much in time

	// check if we crossed a snapshot mark
	bool overMark = true;
	size_type i;
	for (i = pimpl_->nextSnapshotIndex; i < pimpl_->snapshotMarks.size() && overMark; ++i)
	{
		real markTime = pimpl_->snapshotMarks[i]->GetTime();
		// we state that a mark is crossed in two situations: 
		// 1) when the current time mark is almost over the snapshot
		// mark within a certain tolerance, and
		// 2) when we effectively crossed the snapshot mark
		overMark = (abs(currentTime_ - markTime) / lastTimestep_ <= time_tolerance);
		overMark |= (currentTime_ - markTime >= 0);

		// if we have crossed a mark, update internal variables
		if (overMark)
		{
      pimpl_->currentSnapshotMarkTime = markTime;
      pimpl_->hasCrossedSnapshotMark = true;
      pimpl_->nextSnapshotIndex = i;
		}
	}
}

ada::SnapshotMark& ada::AnalysisTimeline::GetSnapshotMark( size_type index ) const
{
	return *pimpl_->snapshotMarks[index];
}
