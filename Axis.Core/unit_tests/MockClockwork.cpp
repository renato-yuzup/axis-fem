#if defined DEBUG || _DEBUG
#include "MockClockwork.hpp"

namespace auc = axis::unit_tests::core;
namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace afm = axis::foundation::memory;

auc::MockClockwork::MockClockwork( real tickAmount )
{
  tickAmount_ = tickAmount;
}

auc::MockClockwork::~MockClockwork( void )
{
  // nothing to do here
}

void auc::MockClockwork::CalculateNextTick( ada::AnalysisTimeline& ti, ada::NumericalModel& model )
{
  ti.NextTimeIncrement() = tickAmount_;
}

void auc::MockClockwork::CalculateNextTick( ada::AnalysisTimeline& ti, ada::NumericalModel& model, real maxTimeIncrement )
{
  ti.NextTimeIncrement() = tickAmount_ > maxTimeIncrement ? maxTimeIncrement : tickAmount_;
}

void auc::MockClockwork::Destroy( void ) const
{
  delete this;
}

bool axis::unit_tests::core::MockClockwork::IsGPUCapable( void ) const
{
  return true;
}

void axis::unit_tests::core::MockClockwork::CalculateNextTickOnGPU( 
  ada::AnalysisTimeline& ti, const afm::RelativePointer&, 
  adal::ExternalSolverFacade& )
{
  ti.NextTimeIncrement() = tickAmount_;
}

void axis::unit_tests::core::MockClockwork::CalculateNextTickOnGPU( 
  ada::AnalysisTimeline& ti, const afm::RelativePointer&, 
  adal::ExternalSolverFacade&, real maxTimeIncrement )
{
  ti.NextTimeIncrement() = tickAmount_ > maxTimeIncrement ? 
                           maxTimeIncrement : tickAmount_;
}

#endif
