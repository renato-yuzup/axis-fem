#pragma once
#include "domain/algorithms/Clockwork.hpp"

namespace axis { namespace unit_tests { namespace core {

class MockClockwork : public axis::domain::algorithms::Clockwork
{
public:
  MockClockwork(real tickAmount);
  virtual ~MockClockwork(void);
  virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, 
    axis::domain::analyses::NumericalModel& model );
  virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, 
    axis::domain::analyses::NumericalModel& model, real maxTimeIncrement );
  virtual void Destroy( void ) const;

  virtual bool IsGPUCapable( void ) const;

  virtual void CalculateNextTickOnGPU( 
    axis::domain::analyses::AnalysisTimeline& timeline, 
    const axis::foundation::memory::RelativePointer&, 
    axis::domain::algorithms::ExternalSolverFacade& );

  virtual void CalculateNextTickOnGPU( 
    axis::domain::analyses::AnalysisTimeline& timeline, 
    const axis::foundation::memory::RelativePointer& , 
    axis::domain::algorithms::ExternalSolverFacade& ,
    real maxTimeIncrement );

private:
  real tickAmount_;
};

} } }
