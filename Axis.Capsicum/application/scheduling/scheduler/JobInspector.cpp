#include "JobInspector.hpp"
#include "domain/algorithms/Solver.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include <iostream>

namespace aass = axis::application::scheduling::scheduler;
namespace ada  = axis::domain::analyses;
namespace adc  = axis::domain::collections;
namespace ade  = axis::domain::elements;
namespace adal = axis::domain::algorithms;

aass::JobInspector::JobInspector( void )
{
  resourceType_ = SchedulerType::kUndefinedScheduler;
}

aass::JobInspector::~JobInspector( void )
{
  // nothing to do here
}

bool aass::JobInspector::operator()( const ada::NumericalModel& model, 
  const adal::Solver& solver )
{
  return Inspect(model, solver);
}

bool aass::JobInspector::Inspect( const ada::NumericalModel& model, 
  const adal::Solver& solver )
{
  bool cpuOk = solver.IsCPUCapable();
  bool gpuOk = solver.IsGPUCapable();

  // ask numerical model for general capability in GPU
  gpuOk &= model.IsGPUCapable();

  // scan each element capability
  adc::ElementSet& allElements = model.Elements();
  size_type elementCount = allElements.Count();
  for (size_type i = 0; i < elementCount && (cpuOk || gpuOk); i++)
  {
    ade::FiniteElement& element = allElements.GetByPosition(i);
    cpuOk &= element.IsCPUCapable();
  }

  // if we can't run in either platform, we have a clash
  if (!(cpuOk || gpuOk))
  {
    resourceType_ = SchedulerType::kUndefinedScheduler;
    return false;
  }

  if (cpuOk && gpuOk) 
  {
    resourceType_ = SchedulerType::kAnyScheduler;
  }
  else if (gpuOk)
  {
    resourceType_ = SchedulerType::kGPUScheduler;
  }
  else
  {
    resourceType_ = SchedulerType::kCPUScheduler;  
  }
  return true;
}

aass::SchedulerType aass::JobInspector::GetJobRequestedResource( void ) const
{
  return resourceType_;
}
