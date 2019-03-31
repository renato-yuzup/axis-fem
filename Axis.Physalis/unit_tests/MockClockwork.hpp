#pragma once
#include "domain/algorithms/Clockwork.hpp"

namespace axis { namespace unit_tests { namespace physalis {

class MockClockwork : public axis::domain::algorithms::Clockwork
{
public:
  MockClockwork(real tickAmount);
  virtual ~MockClockwork(void);

  virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, axis::domain::analyses::NumericalModel& model );

  virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, axis::domain::analyses::NumericalModel& model, real maxTimeIncrement );

  virtual void Destroy( void ) const;
private:
  real tickAmount_;
};

} } }
