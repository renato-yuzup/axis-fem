#include "ExplicitDynamicSolver.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "foundation/memory/pointer.hpp"
#include "domain/algorithms/ExternalSolverFacade.hpp"
#include "domain/analyses/ReducedNumericalModel.hpp"
#include "domain/analyses/ModelOperatorFacade.hpp"
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

bool adal::ExplicitDynamicSolver::DoIsGPUCapable( void ) const
{
  return true;
}

void adal::ExplicitDynamicSolver::StartAnalysisProcessOnGPU( 
  const ada::AnalysisTimeline& timeline, afm::RelativePointer& reducedModelPtr, 
  adal::ExternalSolverFacade& solverFacade )
{
  analysisInfo_ = 
    new ada::TransientAnalysisInfo(timeline.StartTime(), timeline.EndTime());
  const auto& model = absref<ada::ReducedNumericalModel>(reducedModelPtr);
  afm::RelativePointer fR_ptr = model.Dynamics().GetReactionForcePointer();
  afm::RelativePointer fExt_ptr = model.Dynamics().GetExternalLoadsPointer();
  afm::RelativePointer fInt_ptr = model.Dynamics().GetInternalForcesPointer();
  gpuReactionCommand_.SetDynamicVectors(fR_ptr, fExt_ptr, fInt_ptr);
}

void adal::ExplicitDynamicSolver::ExecuteStepOnGPU( 
  const ada::AnalysisTimeline& timeline, afm::RelativePointer& reducedModelPtr, 
  adal::ExternalSolverFacade& solverFacade )
{
  auto& model = absref<ada::ReducedNumericalModel>(reducedModelPtr);
  ada::ModelDynamics& dynamics = model.Dynamics();
  real t = timeline.GetCurrentTimeMark();
  real dt = timeline.NextTimeIncrement();
  real lastDt = timeline.LastTimeIncrement();
  long iterationIndex = timeline.IterationIndex();

  // Run preliminary steps of explicit method
  ExplicitSolverBeforeCommand solverBeforeCmd(reducedModelPtr, vectorMask_, t, 
    dt, iterationIndex);
  solverFacade.RunKernel(solverBeforeCmd);
  solverFacade.Synchronize();

  // update element stress
  ada::ModelOperatorFacade& modelFacade = model.GetOperator();
  modelFacade.UpdateStrain(t, lastDt, dt);
  modelFacade.Synchronize();
  modelFacade.UpdateStress(t, lastDt, dt);
  modelFacade.Synchronize();

  // calculate internal forces for the next step
  afm::RelativePointer internalForcePtr = dynamics.GetInternalForcesPointer();
  modelFacade.CalculateGlobalInternalForce(t, lastDt, dt);
  modelFacade.Synchronize();
  solverFacade.GatherVector(internalForcePtr, reducedModelPtr);
  solverFacade.Synchronize();

  // Run remaining steps of explicit method
  ExplicitSolverAfterCommand solverAfterCmd(reducedModelPtr, _globalLumpedMass, 
    vectorMask_, t, dt, iterationIndex);
  solverFacade.RunKernel(solverAfterCmd);
  solverFacade.Synchronize();

  // Update element geometry
  modelFacade.UpdateGeometry(t, lastDt, dt);
  modelFacade.Synchronize();
}

void adal::ExplicitDynamicSolver::ExitSecondaryStepOnGPU( 
  const ada::AnalysisTimeline& timeline, afm::RelativePointer& reducedModelPtr, 
  adal::ExternalSolverFacade& solverFacade )
{
  // update analysis information
  real t = timeline.GetCurrentTimeMark();
  real dt = timeline.LastTimeIncrement();
  analysisInfo_->SetCurrentAnalysisTime(t);
  analysisInfo_->SetLastTimeStep(dt);
  analysisInfo_->SetIterationIndex(timeline.IterationIndex());
  ada::ReducedNumericalModel& model = 
    absref<ada::ReducedNumericalModel>(reducedModelPtr);
  ada::ModelKinematics& kinematics = model.Kinematics();
  ada::ModelDynamics& dynamics = model.Dynamics();

  // update boundary conditions to next timestep
  solverFacade.UpdateCurves(t);
  solverFacade.UpdateAccelerations(kinematics.GetAccelerationPointer(), t, vectorMask_);  
  solverFacade.UpdateVelocities(kinematics.GetVelocityPointer(), t, vectorMask_);
  solverFacade.UpdateDisplacements(kinematics.GetDisplacementPointer(), t, vectorMask_);
  solverFacade.UpdateExternalLoads(dynamics.GetExternalLoadsPointer(), t, vectorMask_);
  solverFacade.UpdateLocks(kinematics.GetDisplacementPointer(), t, vectorMask_);

  // Calculate reaction force on the next time step
  solverFacade.RunKernel(gpuReactionCommand_);
}

void adal::ExplicitDynamicSolver::ExitPrimaryStepOnGPU( 
  const ada::AnalysisTimeline& timeline, afm::RelativePointer& reducedModelPtr, 
  adal::ExternalSolverFacade& solverFacade )
{
  // notify availability of results
  solverFacade.DispatchMessageAsync(adam::ModelStateUpdateMessage(
    completeNumericalModel_->Kinematics(), completeNumericalModel_->Dynamics()));
}

void adal::ExplicitDynamicSolver::EndAnalysisProcessOnGPU( 
  const ada::AnalysisTimeline& timeline, afm::RelativePointer& reducedModelPtr, 
  adal::ExternalSolverFacade& solverFacade )
{
  // cleanup
  absref<afb::ColumnVector>(_globalLumpedMass).Destroy();
  System::ModelMemory().Deallocate(_globalLumpedMass);
  if (analysisInfo_ != NULL) analysisInfo_->Destroy();
  analysisInfo_ = NULL;
}

void adal::ExplicitDynamicSolver::AllocateGPUData( ada::NumericalModel& model, 
  ada::AnalysisTimeline& )
{
  // Initialize template masks, masked vectors and work vectors
  InitializeModelVectors(model);
  InitializeVectorMask(vectorMask_, model);
  size_type totalDofCount = model.GetTotalDofCount();
  _globalLumpedMass = afb::ColumnVector::Create(totalDofCount);
  absref<afb::ColumnVector>(_globalLumpedMass).ClearAll();

  // store reference to complete model so we can use it in result collection rounds
  completeNumericalModel_ = &model;  
}

size_type adal::ExplicitDynamicSolver::GetGPUThreadsRequired( 
  const ada::NumericalModel& model ) const
{
  return model.GetTotalDofCount();
}

void adal::ExplicitDynamicSolver::PrepareGPUData( ada::NumericalModel& model, 
  ada::AnalysisTimeline& timeline )
{
  // 1) Initialize global lumped mass matrix
  BuildGlobalMassMatrix(_globalLumpedMass, model);

  // 2) Set initial conditions (on CPU)
  UpdateBoundaryConditions(model, timeline.GetCurrentTimeMark(), 
    timeline.LastTimeIncrement(), absptr<char>(vectorMask_));
}
