#if defined DEBUG || _DEBUG
#include "MockClockwork.hpp"

namespace aup = axis::unit_tests::orange;
namespace ada = axis::domain::analyses;

aup::MockClockwork::MockClockwork( real tickAmount )
{
  tickAmount_ = tickAmount;
}

aup::MockClockwork::~MockClockwork( void )
{
  // nothing to do here
}

void aup::MockClockwork::CalculateNextTick( ada::AnalysisTimeline& ti, ada::NumericalModel& model )
{
  ti.NextTimeIncrement() = tickAmount_;
}

void aup::MockClockwork::CalculateNextTick( ada::AnalysisTimeline& ti, ada::NumericalModel& model, real maxTimeIncrement )
{
  ti.NextTimeIncrement() = tickAmount_ > maxTimeIncrement ? maxTimeIncrement : tickAmount_;
}

void aup::MockClockwork::Destroy( void ) const
{
  delete this;
}

#endif
