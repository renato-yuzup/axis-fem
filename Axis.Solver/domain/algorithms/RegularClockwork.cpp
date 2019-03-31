#include "RegularClockwork.hpp"

namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace afm = axis::foundation::memory;

adal::RegularClockwork::RegularClockwork( real timeIncrement )
{
	_timestepIncrement = timeIncrement;
}

adal::RegularClockwork::~RegularClockwork( void )
{
	// nothing to do here
}

void adal::RegularClockwork::CalculateNextTick( ada::AnalysisTimeline& ti, ada::NumericalModel& )
{
	ti.NextTimeIncrement() = _timestepIncrement;
}

void adal::RegularClockwork::CalculateNextTick( ada::AnalysisTimeline& ti, ada::NumericalModel&, 
                                                real maxTimeIncrement )
{
	ti.NextTimeIncrement() = _timestepIncrement > maxTimeIncrement ? maxTimeIncrement : _timestepIncrement;
}

void adal::RegularClockwork::Destroy( void ) const
{
	delete this;
}

adal::Clockwork& adal::RegularClockwork::Create( real timeIncrement )
{
	return *new RegularClockwork(timeIncrement);
}

bool adal::RegularClockwork::IsGPUCapable( void ) const
{
  return true;
}

void adal::RegularClockwork::CalculateNextTickOnGPU( 
  ada::AnalysisTimeline& timeline, const afm::RelativePointer&,
  adal::ExternalSolverFacade& )
{
  timeline.NextTimeIncrement() = _timestepIncrement;
}

void adal::RegularClockwork::CalculateNextTickOnGPU( 
  ada::AnalysisTimeline& timeline, const afm::RelativePointer&,
  adal::ExternalSolverFacade&, real maxTimeIncrement )
{
  timeline.NextTimeIncrement() = _timestepIncrement > maxTimeIncrement ? 
                                 maxTimeIncrement : _timestepIncrement;
}
