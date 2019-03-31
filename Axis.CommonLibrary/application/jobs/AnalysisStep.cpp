#include "AnalysisStep.hpp"
#include "AnalysisStep_Pimpl.hpp"
#include "foundation/NotImplementedException.hpp"

namespace aao = axis::application::output;
namespace aaj = axis::application::jobs;
namespace aapp = axis::application::post_processing;
namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;

aaj::AnalysisStep::~AnalysisStep( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

aaj::AnalysisStep& aaj::AnalysisStep::Create( real startTime, real endTime, adal::Solver& solver,
                                              aao::ResultBucket& bucket)
{
  return *new AnalysisStep(startTime, endTime, solver, bucket);
}

void aaj::AnalysisStep::Destroy( void ) const
{
  delete this;
}

adal::Solver& aaj::AnalysisStep::GetSolver( void )
{
	return pimpl_->solver;
}

const adal::Solver& aaj::AnalysisStep::GetSolver( void ) const
{
  return pimpl_->solver;
}

const aapp::PostProcessor& aaj::AnalysisStep::GetPostProcessor( void ) const
{
  return pimpl_->postProcessor;
}

aapp::PostProcessor& aaj::AnalysisStep::GetPostProcessor( void )
{
  return pimpl_->postProcessor;
}

const ada::AnalysisTimeline& aaj::AnalysisStep::GetTimeline( void ) const
{
  return pimpl_->timeline;
}

ada::AnalysisTimeline& aaj::AnalysisStep::GetTimeline( void )
{
  return pimpl_->timeline;
}

real aaj::AnalysisStep::GetStartTime( void ) const
{
	return pimpl_->startTime;
}

real aaj::AnalysisStep::GetEndTime( void ) const
{
	return pimpl_->endTime;
}

const aao::ResultBucket& aaj::AnalysisStep::GetResults( void ) const
{
  return pimpl_->resultBucket;
}

aao::ResultBucket& aaj::AnalysisStep::GetResults( void )
{
  return pimpl_->resultBucket;
}

const adc::BoundaryConditionCollection& aaj::AnalysisStep::NodalLoads( void ) const
{
	return pimpl_->nodalLoads;
}

adc::BoundaryConditionCollection& aaj::AnalysisStep::NodalLoads( void )
{
	return pimpl_->nodalLoads;
}

const adc::BoundaryConditionCollection& aaj::AnalysisStep::Accelerations( void ) const
{
	return pimpl_->accelerations;
}

adc::BoundaryConditionCollection& aaj::AnalysisStep::Accelerations( void )
{
	return pimpl_->accelerations;
}

const adc::BoundaryConditionCollection& aaj::AnalysisStep::Velocities( void ) const
{
	return pimpl_->velocities;
}

adc::BoundaryConditionCollection& aaj::AnalysisStep::Velocities( void )
{
	return pimpl_->velocities;
}

const adc::BoundaryConditionCollection& aaj::AnalysisStep::Displacements( void ) const
{
	return pimpl_->displacements;
}

adc::BoundaryConditionCollection& aaj::AnalysisStep::Displacements( void )
{
	return pimpl_->displacements;
}

const adc::BoundaryConditionCollection& aaj::AnalysisStep::Locks( void ) const
{
	return pimpl_->locks;
}

adc::BoundaryConditionCollection& aaj::AnalysisStep::Locks( void )
{
	return pimpl_->locks;
}

bool aaj::AnalysisStep::DefinesBoundaryCondition( ade::DoF& dof ) const
{
	return pimpl_->displacements.Contains(dof) ||
         pimpl_->nodalLoads.Contains(dof) ||
         pimpl_->accelerations.Contains(dof) ||
         pimpl_->velocities.Contains(dof) ||
         pimpl_->locks.Contains(dof);
}

axis::String aaj::AnalysisStep::GetName( void ) const
{
	return pimpl_->name;
}

void aaj::AnalysisStep::SetName( const axis::String& name )
{
	pimpl_->name = name;
}

aaj::AnalysisStep::AnalysisStep( real startTime, real endTime, 
                                 adal::Solver& solver, aao::ResultBucket& bucket )
{
  ada::AnalysisTimeline& tl = ada::AnalysisTimeline::Create(startTime, endTime);
  pimpl_ = new Pimpl(solver, tl, bucket);
  pimpl_->startTime = startTime;
  pimpl_->endTime = endTime;
}
