#if defined DEBUG || defined _DEBUG

#pragma once
#include "domain/algorithms/Solver.hpp"
#include "MockClockwork.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"

/**************************************************************************************************
 * <summary>	Just a mock solver to use in our test cases. </summary>
 **************************************************************************************************/
class MockSolver : public axis::domain::algorithms::Solver
{
public:
	MockSolver(axis::domain::algorithms::Clockwork& cw);
	~MockSolver(void);
	virtual void Destroy( void ) const;
	virtual int GetSolverEventSourceId( void ) const;
	virtual axis::services::diagnostics::information::SolverCapabilities GetCapabilities( void ) const;
  int GetPrimaryStepCount(void) const;
  int GetSecondaryStepCount(void) const;
  int GetMicroStepsCount(void) const;
  bool IsStepNestingOk(void) const;
private:
	virtual void StartAnalysisProcess( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                     axis::domain::analyses::NumericalModel& model );
	virtual void EnterPrimaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                 axis::domain::analyses::NumericalModel& model );
	virtual void EnterSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                   axis::domain::analyses::NumericalModel& model );
	virtual void ExecuteStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                            axis::domain::analyses::NumericalModel& model );
	virtual void ExitSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                  axis::domain::analyses::NumericalModel& model );
	virtual void ExitPrimaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                axis::domain::analyses::NumericalModel& model );
	virtual void EndAnalysisProcess( const axis::domain::analyses::AnalysisTimeline& timeline, 
                                   axis::domain::analyses::NumericalModel& model );
  virtual const axis::domain::analyses::AnalysisInfo& GetAnalysisInformation( void ) const;

  virtual bool DoIsGPUCapable( void ) const;

  virtual void StartAnalysisProcessOnGPU( 
    const axis::domain::analyses::AnalysisTimeline& timeline, 
    axis::foundation::memory::RelativePointer& reducedModelPtr, 
    axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  virtual void EnterPrimaryStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, axis::foundation::memory::RelativePointer& reducedModelPtr, axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  virtual void EnterSecondaryStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, axis::foundation::memory::RelativePointer& reducedModelPtr, axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  virtual void ExecuteStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, axis::foundation::memory::RelativePointer& reducedModelPtr, axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  virtual void ExitSecondaryStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, axis::foundation::memory::RelativePointer& reducedModelPtr, axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  virtual void ExitPrimaryStepOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, axis::foundation::memory::RelativePointer& reducedModelPtr, axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  virtual void EndAnalysisProcessOnGPU( const axis::domain::analyses::AnalysisTimeline& timeline, axis::foundation::memory::RelativePointer& reducedModelPtr, axis::domain::algorithms::ExternalSolverFacade& solverFacade );

  int _state;
  bool _analysisStarted;
  int _enterPrimaryCount, _auxEnterPrimaryCount;
  int _enterSecondaryCount, _auxEnterSecondaryCount;
  int _microStepsCount;
  axis::domain::analyses::TransientAnalysisInfo *info_;
};

#endif