#if defined DEBUG || defined _DEBUG

#include "MockSolver.hpp"
#include "unit_tests.hpp"

namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace aup = axis::unit_tests::physalis;
namespace asdi = axis::services::diagnostics::information;

aup::MockSolver::MockSolver(adal::Clockwork& cw) : adal::Solver(cw)
{
	_state = 0;
	_analysisStarted = false;
	_auxEnterSecondaryCount = 0;
	_auxEnterPrimaryCount = 0;
	_enterPrimaryCount = 0;
	_enterSecondaryCount = 0;
	_microStepsCount = 0;
  info_ = NULL;
}


aup::MockSolver::~MockSolver(void)
{
  if (info_ != NULL) info_->Destroy();
  info_ = NULL;
}

void aup::MockSolver::Destroy( void ) const
{
	delete this;
}

int aup::MockSolver::GetSolverEventSourceId( void ) const
{
	return 0;
}

asdi::SolverCapabilities aup::MockSolver::GetCapabilities( void ) const
{
	return asdi::SolverCapabilities(_T("MockSolver"), _T(""), false, false, false, false);
}

void aup::MockSolver::StartAnalysisProcess( const ada::AnalysisTimeline& timeline, 
                                            ada::NumericalModel& model )
{
	if (_state != 0)
	{
		Assert::Fail(_T("StartAnalysisProces() was not the first method called on run-time."));
	}
  info_ = new ada::TransientAnalysisInfo(timeline.StartTime(), timeline.EndTime());
	_state = 1;
	_analysisStarted = true;
}

void aup::MockSolver::EnterPrimaryStep( const ada::AnalysisTimeline& timeline, 
                                        ada::NumericalModel& model )
{
	if (!_analysisStarted)
	{
		Assert::Fail(_T("EnterPrimaryStep() called before starting analysis."));
	}
	_state = 2;
	_enterPrimaryCount++; _auxEnterPrimaryCount++;
	if (_auxEnterPrimaryCount > 1)
	{
		Assert::Fail(_T("Nested EnterPrimaryStep() calls."));
	}
}

void aup::MockSolver::EnterSecondaryStep( const ada::AnalysisTimeline& timeline, 
                                          ada::NumericalModel& model )
{
	if (!_analysisStarted)
	{
		Assert::Fail(_T("EnterSecondaryStep() called before starting analysis."));
	}
	if (_state != 2 && _state != 3 && _state != 1)
	{
		Assert::Fail(_T("EnterSecondaryStep() called before entering a primary step or terminating execution."));
	}
	_state = 3;
	_enterSecondaryCount++; _auxEnterSecondaryCount++;
	if (_auxEnterSecondaryCount > 1)
	{
		Assert::Fail(_T("Nested EnterSecondaryStep() calls."));
	}
}

void aup::MockSolver::ExecuteStep( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	if (!_analysisStarted)
	{
		Assert::Fail(_T("ExecuteStep() called before starting analysis."));
	}
	if (_state != 3)
	{
		Assert::Fail(_T("ExecuteStep() called before entering secondary step."));
	}
	_state = 4;
	_microStepsCount++;
}

void aup::MockSolver::ExitSecondaryStep( const ada::AnalysisTimeline& timeline, 
                                         ada::NumericalModel& model )
{
	if (!_analysisStarted)
	{
		Assert::Fail(_T("ExitSecondaryStep() called before starting analysis."));
	}
	if (_state != 4)
	{
		Assert::Fail(_T("ExitSecondaryStep() called before terminating step execution."));
	}
	_state = 3;
	_auxEnterSecondaryCount--;
	if (_auxEnterSecondaryCount > 1)
	{
		Assert::Fail(_T("Nested ExitSecondaryStep() calls."));
	}
}

void aup::MockSolver::ExitPrimaryStep( const ada::AnalysisTimeline& timeline, 
                                       ada::NumericalModel& model )
{
	if (!_analysisStarted)
	{
		Assert::Fail(_T("ExitPrimaryStep() called before starting analysis."));
	}
	if (_state != 3)
	{
		Assert::Fail(_T("ExitPrimaryStep() called before exiting secondary step."));
	}
	_state = 2;
	_auxEnterPrimaryCount--;
	if (_auxEnterPrimaryCount > 1)
	{
		Assert::Fail(_T("Nested ExitPrimaryStep() calls."));
	}
}

void aup::MockSolver::EndAnalysisProcess( const ada::AnalysisTimeline& timeline, 
                                          ada::NumericalModel& model )
{
	if (!_analysisStarted)
	{
		Assert::Fail(_T("EndAnalysisProcess() called before starting analysis."));
	}
	if (_state != 2 && _state != 3)
	{
		Assert::Fail(_T("EndAnalysisProcess() called before exiting primary step."));
	}
	_state = 0;
	_analysisStarted = false;
}

int aup::MockSolver::GetPrimaryStepCount( void ) const
{
	return _enterPrimaryCount;
}

int aup::MockSolver::GetSecondaryStepCount( void ) const
{
	return _enterSecondaryCount;
}

bool aup::MockSolver::IsStepNestingOk( void ) const
{
	return _auxEnterPrimaryCount == 0 && _auxEnterSecondaryCount == 0;
}

int aup::MockSolver::GetMicroStepsCount( void ) const
{
	return _microStepsCount;
}

const ada::AnalysisInfo& aup::MockSolver::GetAnalysisInformation( void ) const
{
  return *info_;
}

#endif

