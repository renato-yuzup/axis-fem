#include "ExplicitDynamicSolver.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "foundation/memory/pointer.hpp"
#include "domain/algorithms/ExternalSolverFacade.hpp"
#include "domain/analyses/ReducedNumericalModel.hpp"
#include "domain/analyses/ModelOperatorFacade.hpp"
#include "dof_status.hpp"
#include "ExplicitDynamicSolver_Helper.hpp"

namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace adam = adal::messages;
namespace adb = axis::domain::boundary_conditions;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace asdi = axis::services::diagnostics::information;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

adal::ExplicitDynamicSolver::ExplicitDynamicSolver(adal::Clockwork& clockwork) :
  Solver(clockwork)
{
  analysisInfo_ = NULL;
}

adal::ExplicitDynamicSolver::~ExplicitDynamicSolver( void )
{
  if (analysisInfo_ != NULL) analysisInfo_->Destroy();
  analysisInfo_ = NULL;
}

void adal::ExplicitDynamicSolver::StartAnalysisProcess( 
  const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
  analysisInfo_ = new ada::TransientAnalysisInfo(timeline.StartTime(), 
    timeline.EndTime());
	// 1) Initialize template masks, masked vectors and work vectors
	InitializeVectorMask(vectorMask_, model);
	InitializeModelVectors(model);
  vecMask_ = absptr<char>(vectorMask_);

  // 2) Initialize global lumped mass matrix
  size_type totalDofCount = model.GetTotalDofCount();
  _globalLumpedMass = afb::ColumnVector::Create(totalDofCount);
	BuildGlobalMassMatrix(_globalLumpedMass, model);
	 
	// 3) Set initial conditions
	UpdateBoundaryConditions(model, timeline.GetCurrentTimeMark(), 
    timeline.LastTimeIncrement(), vecMask_);
}

void adal::ExplicitDynamicSolver::ExecuteStep( 
  const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	/*****************************************************************************
	/*         CENTRAL DIFFERENCE METHOD ALGORITHM MAIN LOOP
	*****************************************************************************/
	// initialize work vectors
  ada::ModelDynamics& dynamics     = model.Dynamics();
  ada::ModelKinematics& kinematics = model.Kinematics();
  afb::ColumnVector& u    = kinematics.Displacement();
  afb::ColumnVector& v    = kinematics.Velocity();
  afb::ColumnVector& a    = kinematics.Acceleration();
  afb::ColumnVector& du   = kinematics.DisplacementIncrement();
  afb::ColumnVector& fR   = dynamics.ReactionForce();
  afb::ColumnVector& fInt = dynamics.InternalForces();
  afb::ColumnVector& fExt = dynamics.ExternalLoads();
  afb::ColumnVector& m    = absref<afb::ColumnVector>(_globalLumpedMass);
  const char *dofMask     = vecMask_;

  // initialize constants
  const real dt = timeline.NextTimeIncrement();
  const size_type dofCount = (size_type)model.GetTotalDofCount();

  #pragma omp parallel for
  for (size_type index = 0; index < dofCount; ++index)
  {
    // 1) Calculate next displacement increment and update displacement
    if (dofMask[index] == DOF_STATUS_FREE || dofMask[index] == DOF_STATUS_PRESCRIBED_VELOCITY)
    {
      du(index) = v(index) * dt;
    }
    u(index) += du(index);
  }

  // 2) Update node coordinates
  UpdateNodeCoordinates(model, du);

  // 3) Update element stress
  UpdateModelStressState(model, du, v, timeline);

  // 4) Calculate internal forces
  CalculateInternalForce(fInt, model, u, du, v, timeline);

  // 5) Calculate effective nodal force
  afb::VectorSum(fR, +1.0, fExt, +1.0, fInt);

  #pragma omp parallel for
  for (size_type index = 0; index < dofCount; ++index)
  {
    // 6) Calculate accelerations
    if (dofMask[index] != DOF_STATUS_LOCKED && dofMask[index] != DOF_STATUS_PRESCRIBED_VELOCITY)
    {
      a(index) = fR(index) / m(index);
    }

    // 7) Update velocities
    if (dofMask[index] == DOF_STATUS_FREE || dofMask[index] == DOF_STATUS_PRESCRIBED_DISPLACEMENT)
    {
      v(index) += a(index) * dt;
    }
  }

  // 8) Update geometry information
  UpdateGeometryInformation(model);
}

void adal::ExplicitDynamicSolver::ExitSecondaryStep( 
  const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	ada::ModelDynamics& dynamics = model.Dynamics();
	real t = timeline.GetCurrentTimeMark();
  real dt = timeline.LastTimeIncrement();

  // update analysis information
  analysisInfo_->SetCurrentAnalysisTime(t);
  analysisInfo_->SetLastTimeStep(dt);
  analysisInfo_->SetIterationIndex(timeline.IterationIndex());
	UpdateBoundaryConditions(model, t, dt, vecMask_);
}

void adal::ExplicitDynamicSolver::ExitPrimaryStep( 
  const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	// notify availability of results
	DispatchMessage(adam::ModelStateUpdateMessage(model.Kinematics(), 
    model.Dynamics()));
}

void adal::ExplicitDynamicSolver::EndAnalysisProcess( 
  const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	// destroy mask vector
	System::ModelMemory().Deallocate(vectorMask_);

	// destroy work vectors
	absref<afb::ColumnVector>(_globalLumpedMass).Destroy();
  System::ModelMemory().Deallocate(_globalLumpedMass);

  if (analysisInfo_ != NULL) analysisInfo_->Destroy();
  analysisInfo_ = NULL;
}

void adal::ExplicitDynamicSolver::Destroy( void ) const
{
	delete this;
}

int adal::ExplicitDynamicSolver::GetSolverEventSourceId( void ) const
{
	return 0x72;
}

asdi::SolverCapabilities adal::ExplicitDynamicSolver::GetCapabilities(void) const
{
	return asdi::SolverCapabilities(
		_T("EXPLICIT STANDARD SOLVER"), 
		_T("Solver for time-dependent problems using explicit time integration procedure."), 
		true, true, true, false);
}

const ada::AnalysisInfo& adal::ExplicitDynamicSolver::GetAnalysisInformation( 
  void) const
{
  return *analysisInfo_;
}

