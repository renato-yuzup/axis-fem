#pragma once
#include "domain/algorithms/Solver.hpp"
#include "foundation/blas/VectorView.hpp"
#include "foundation/blas/SubColumnVector.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"
#include "ExplicitSolverAfterCommand.hpp"
#include "ExplicitSolverBeforeCommand.hpp"
#include "UpdateReactionForceCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace algorithms {
class ExplicitDynamicSolver : public Solver
{
public:
  ExplicitDynamicSolver(axis::domain::algorithms::Clockwork& clockwork);
  ~ExplicitDynamicSolver(void);
  virtual void Destroy( void ) const;
  virtual int GetSolverEventSourceId( void ) const;
  virtual axis::services::diagnostics::information::SolverCapabilities GetCapabilities( void ) const;
  virtual const axis::domain::analyses::AnalysisInfo& GetAnalysisInformation( void ) const;
  virtual void AllocateGPUData( axis::domain::analyses::NumericalModel& model, 
                                axis::domain::analyses::AnalysisTimeline& timeline );
  virtual size_type GetGPUThreadsRequired( const axis::domain::analyses::NumericalModel& model ) const;
  virtual void PrepareGPUData( axis::domain::analyses::NumericalModel& model, 
                               axis::domain::analyses::AnalysisTimeline& timeline );
private:
	virtual void StartAnalysisProcess( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                      axis::domain::analyses::NumericalModel& model );
	virtual void ExecuteStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                            axis::domain::analyses::NumericalModel& model );
	virtual void ExitSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                  axis::domain::analyses::NumericalModel& model );
	virtual void ExitPrimaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                axis::domain::analyses::NumericalModel& model );
	virtual void EndAnalysisProcess( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                    axis::domain::analyses::NumericalModel& model );

  virtual bool DoIsGPUCapable( void ) const;
  virtual void StartAnalysisProcessOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                          axis::foundation::memory::RelativePointer& reducedModelPtr, 
                                          axis::domain::algorithms::ExternalSolverFacade& solverFacade );
  virtual void ExecuteStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                 axis::foundation::memory::RelativePointer& reducedModelPtr, 
                                 axis::domain::algorithms::ExternalSolverFacade& solverFacade );
  virtual void ExitSecondaryStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                       axis::foundation::memory::RelativePointer& reducedModelPtr, 
                                       axis::domain::algorithms::ExternalSolverFacade& solverFacade );
  virtual void ExitPrimaryStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                     axis::foundation::memory::RelativePointer& reducedModelPtr, 
                                     axis::domain::algorithms::ExternalSolverFacade& solverFacade );
  virtual void EndAnalysisProcessOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                        axis::foundation::memory::RelativePointer& reducedModelPtr, 
                                        axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  axis::foundation::memory::RelativePointer vectorMask_;
  char *vecMask_;
  UpdateReactionForceCommand gpuReactionCommand_;
  axis::domain::analyses::NumericalModel *completeNumericalModel_;

  // these are the real vectors
  axis::foundation::memory::RelativePointer _globalLumpedMass;

  axis::domain::analyses::TransientAnalysisInfo *analysisInfo_;
};

} } } // namespace axis::domain::algorithms

