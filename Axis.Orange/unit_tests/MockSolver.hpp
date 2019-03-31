#pragma once
#include "domain/algorithms/Solver.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"

/**************************************************************************************************
 * <summary>	Just a mock solver to use in our test cases. </summary>
 **************************************************************************************************/
namespace axis { namespace unit_tests { namespace orange {

class MockSolver : public axis::domain::algorithms::Solver
{
private:
	int _state;
	bool _analysisStarted;
	int _enterPrimaryCount, _auxEnterPrimaryCount;
	int _enterSecondaryCount, _auxEnterSecondaryCount;
	int _microStepsCount;
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

  axis::domain::analyses::TransientAnalysisInfo *info_;
};

} } }