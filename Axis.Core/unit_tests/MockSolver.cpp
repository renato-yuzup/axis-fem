#if defined DEBUG || defined _DEBUG

#include "MockSolver.hpp"
#include "unit_tests/unit_tests.hpp"

namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace asdi = axis::services::diagnostics::information;
namespace afm = axis::foundation::memory;

MockSolver::MockSolver(adal::Clockwork& cw) : adal::Solver(cw)
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


MockSolver::~MockSolver(void)
{
  if (info_ != NULL) info_->Destroy();
  info_ = NULL;
}

void MockSolver::Destroy( void ) const
{
	delete this;
}

int MockSolver::GetSolverEventSourceId( void ) const
{
	return 0;
}

asdi::SolverCapabilities MockSolver::GetCapabilities( void ) const
{
	return asdi::SolverCapabilities(_T("MockSolver"), _T(""), false, false, false, false);
}

void MockSolver::StartAnalysisProcess( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	if (_state != 0)
	{
		Assert::Fail(_T("StartAnalysisProces() was not the first method called on run-time."));
	}
  info_ = new ada::TransientAnalysisInfo(timeline.StartTime(), timeline.EndTime());
	_state = 1;
	_analysisStarted = true;
}

void MockSolver::EnterPrimaryStep( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
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

void MockSolver::EnterSecondaryStep( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
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

void MockSolver::ExecuteStep( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
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

void MockSolver::ExitSecondaryStep( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
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

void MockSolver::ExitPrimaryStep( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
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

void MockSolver::EndAnalysisProcess( const ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
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

int MockSolver::GetPrimaryStepCount( void ) const
{
	return _enterPrimaryCount;
}

int MockSolver::GetSecondaryStepCount( void ) const
{
	return _enterSecondaryCount;
}

bool MockSolver::IsStepNestingOk( void ) const
{
	return _auxEnterPrimaryCount == 0 && _auxEnterSecondaryCount == 0;
}

int MockSolver::GetMicroStepsCount( void ) const
{
	return _microStepsCount;
}

const ada::AnalysisInfo& MockSolver::GetAnalysisInformation( void ) const
{
  return *info_;
}

bool MockSolver::DoIsGPUCapable( void ) const
{
  return true;
}

void MockSolver::StartAnalysisProcessOnGPU( 
  const ada::AnalysisTimeline& timeline, afm::RelativePointer&, 
  adal::ExternalSolverFacade& )
{
  if (_state != 0)
  {
    Assert::Fail(_T("StartAnalysisProces() was not the first method called on run-time."));
  }
  info_ = new ada::TransientAnalysisInfo(timeline.StartTime(), timeline.EndTime());
  info_->SetCurrentAnalysisTime(timeline.StartTime());
  _state = 1;
  _analysisStarted = true;
}

void MockSolver::EnterPrimaryStepOnGPU( const ada::AnalysisTimeline&, 
  afm::RelativePointer&, adal::ExternalSolverFacade& )
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

void MockSolver::EnterSecondaryStepOnGPU( const ada::AnalysisTimeline&, 
  afm::RelativePointer&, adal::ExternalSolverFacade& )
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

void MockSolver::ExecuteStepOnGPU( const ada::AnalysisTimeline&, 
  afm::RelativePointer&, adal::ExternalSolverFacade& )
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

void MockSolver::ExitSecondaryStepOnGPU( const ada::AnalysisTimeline&, 
  afm::RelativePointer&, adal::ExternalSolverFacade& )
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

void MockSolver::ExitPrimaryStepOnGPU( const ada::AnalysisTimeline&, 
  afm::RelativePointer&, adal::ExternalSolverFacade& )
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

void MockSolver::EndAnalysisProcessOnGPU( const ada::AnalysisTimeline&, 
  afm::RelativePointer&, adal::ExternalSolverFacade& )
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

#endif