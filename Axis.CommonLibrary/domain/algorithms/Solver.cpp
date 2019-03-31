#include "Solver.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "services/messaging/LogMessage.hpp"
#include "domain/algorithms/messages/SnapshotStartMessage.hpp"
#include "domain/algorithms/messages/SnapshotEndMessage.hpp"
#include "domain/analyses/AnalysisInfo.hpp"
#include "domain/analyses/ReducedNumericalModel.hpp"
#include "domain/analyses/ModelOperatorFacade.hpp"
#include "ExternalSolverFacade.hpp"

namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace adalm = adal::messages;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

adal::Solver::Solver( Clockwork& clockwork ) : _clock(clockwork)
{
	_mute = false; _debug = false; _verbose = false;
}

adal::Solver::~Solver( void )
{
	_clock.Destroy();
}

void adal::Solver::LogSolverMessage( const axis::String& message ) const
{
	if (_mute || !_verbose) return;
	axis::String name = GetCapabilities().GetSolverName();
	axis::String msg = name + _T(": ") + message;
	asmm::LogMessage logMsg(msg);
	logMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));
	DispatchMessage(logMsg);
}

void adal::Solver::LogSolverInfoMessage( int eventId, const axis::String& message ) const
{
	if (_mute) return;
	axis::String name = GetCapabilities().GetSolverName();
	axis::String msg = name + _T(": ") + message;
	asmm::InfoMessage infoMsg(eventId, msg);
	infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));
	DispatchMessage(infoMsg);
}

void adal::Solver::LogSolverErrorMessage( int eventId, const axis::String& message ) const
{
	if (_mute) return;
	axis::String name = GetCapabilities().GetSolverName();
	axis::String msg = name + _T(": ") + message;
	asmm::ErrorMessage errorMsg(eventId, msg);
	errorMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));
	DispatchMessage(errorMsg);
}

void adal::Solver::LogSolverWarningMessage( int eventId, const axis::String& message ) const
{
	if (_mute) return;
	axis::String name = GetCapabilities().GetSolverName();
	axis::String msg = name + _T(": ") + message;
	asmm::WarningMessage warningMsg(eventId, msg);
	warningMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));
	DispatchMessage(warningMsg);
}

void adal::Solver::LogDebugMatrix( const afb::DenseMatrix& matrix, const axis::String& matrixName ) const
{
	// ignore if debug is not enabled
	if (!IsDebugEnabled()) return;

	size_type rows = matrix.Rows();
	size_type cols = matrix.Columns();

	axis::String name = GetCapabilities().GetSolverName();

	axis::String line = matrixName + _T(" = [ ");
	for (size_type row = 0; row < rows; ++row)
	{
		for (size_type col = 0; col < cols; ++col)
		{
			line += axis::String::double_parse((real)matrix.GetElement(row, col)) + _T(" ");
		}
		if (row < rows -1)
		{
			line += _T(";");
		}
		else
		{
			line += _T("]");
		}
		asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
		infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
		DispatchMessage(infoMsg);
		line.clear();
	}
}

void adal::Solver::LogDebugMatrix( const afb::SymmetricMatrix& matrix, const axis::String& matrixName ) const
{
  // ignore if debug is not enabled
  if (!IsDebugEnabled()) return;

  size_type rows = matrix.Rows();
  size_type cols = matrix.Columns();

  axis::String name = GetCapabilities().GetSolverName();

  axis::String line = matrixName + _T(" = [ ");
  for (size_type row = 0; row < rows; ++row)
  {
    for (size_type col = 0; col < cols; ++col)
    {
      line += axis::String::double_parse((real)matrix.GetElement(row, col)) + _T(" ");
    }
    if (row < rows -1)
    {
      line += _T(";");
    }
    else
    {
      line += _T("]");
    }
    asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
    infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
    DispatchMessage(infoMsg);
    line.clear();
  }
}

void adal::Solver::LogDebugMatrix( const afb::LowerTriangularMatrix& matrix, const axis::String& matrixName ) const
{
  // ignore if debug is not enabled
  if (!IsDebugEnabled()) return;

  size_type rows = matrix.Rows();
  size_type cols = matrix.Columns();

  axis::String name = GetCapabilities().GetSolverName();

  axis::String line = matrixName + _T(" = [ ");
  for (size_type row = 0; row < rows; ++row)
  {
    for (size_type col = 0; col < cols; ++col)
    {
      line += axis::String::double_parse((real)matrix.GetElement(row, col)) + _T(" ");
    }
    if (row < rows -1)
    {
      line += _T(";");
    }
    else
    {
      line += _T("]");
    }
    asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
    infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
    DispatchMessage(infoMsg);
    line.clear();
  }
}

void adal::Solver::LogDebugMatrix( const afb::UpperTriangularMatrix& matrix, const axis::String& matrixName ) const
{
  // ignore if debug is not enabled
  if (!IsDebugEnabled()) return;

  size_type rows = matrix.Rows();
  size_type cols = matrix.Columns();

  axis::String name = GetCapabilities().GetSolverName();

  axis::String line = matrixName + _T(" = [ ");
  for (size_type row = 0; row < rows; ++row)
  {
    for (size_type col = 0; col < cols; ++col)
    {
      line += axis::String::double_parse((real)matrix.GetElement(row, col)) + _T(" ");
    }
    if (row < rows -1)
    {
      line += _T(";");
    }
    else
    {
      line += _T("]");
    }
    asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
    infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
    DispatchMessage(infoMsg);
    line.clear();
  }
}

void adal::Solver::LogDebugVector( const afb::ColumnVector& vector, 
                                   const axis::String& vectorName, 
                                   bool asRowVector /*= false*/ ) const
{
	// ignore if debug is not enabled
	if (!IsDebugEnabled()) return;

	size_type size = vector.Rows();
	axis::String name = GetCapabilities().GetSolverName();
	axis::String line = vectorName + _T(" = [ ");
	for (size_type i = 0; i < size; ++i)
	{
    line += String::double_parse((real)vector(i));
		if (asRowVector)
		{
			line += _T(" ");
		}
		else
		{
			if (i < size - 1)
			{
				line += _T(";");
			}
			else
			{
				line += _T("]");
			}
			asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
			infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
			DispatchMessage(infoMsg);
			line.clear();
		}
	}
	if (asRowVector)
	{
		line += _T("]");
		asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
		infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
		DispatchMessage(infoMsg);
		line.clear();
	}
}

void adal::Solver::LogDebugVector( const afb::RowVector& vector, 
                                  const axis::String& vectorName, 
                                  bool asColVector /*= false*/ ) const
{
  // ignore if debug is not enabled
  if (!IsDebugEnabled()) return;

  size_type size = vector.Rows();
  axis::String name = GetCapabilities().GetSolverName();
  axis::String line = vectorName + _T(" = [ ");
  for (size_type i = 0; i < size; ++i)
  {
    line += String::double_parse((real)vector(i));
    if (!asColVector)
    {
      line += _T(" ");
    }
    else
    {
      if (i < size - 1)
      {
        line += _T(";");
      }
      else
      {
        line += _T("]");
      }
      asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
      infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
      DispatchMessage(infoMsg);
      line.clear();
    }
  }
  if (!asColVector)
  {
    line += _T("]");
    asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
    infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
    DispatchMessage(infoMsg);
    line.clear();
  }
}

void adal::Solver::LogDebugScalar( real value, const axis::String& scalarName ) const
{
	// ignore if debug is not enabled
	if (!IsDebugEnabled()) return;

	axis::String name = GetCapabilities().GetSolverName();
	axis::String line = scalarName + _T(" = ");
	line += axis::String::double_parse((real)value);
	asmm::InfoMessage infoMsg(0, line, asmm::InfoMessage::InfoDebugLevel);
	infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
	DispatchMessage(infoMsg);
}

void adal::Solver::LogDebugMessage( const axis::String& message ) const
{
	// ignore if debug is not enabled
	if (!IsDebugEnabled()) return;
	axis::String name = GetCapabilities().GetSolverName();
	asmm::InfoMessage infoMsg(0, message, asmm::InfoMessage::InfoDebugLevel);
	infoMsg.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(GetSolverEventSourceId(), name));		
	DispatchMessage(infoMsg);
}

void adal::Solver::StartAnalysisProcess( const ada::AnalysisTimeline& timeline, 
                                         ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

void adal::Solver::EnterPrimaryStep( const ada::AnalysisTimeline& timeline, 
                                     ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

void adal::Solver::EnterSecondaryStep( const ada::AnalysisTimeline& timeline, 
                                       ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

void adal::Solver::ExecuteStep( const ada::AnalysisTimeline& timeline, 
                                ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

void adal::Solver::ExitSecondaryStep( const ada::AnalysisTimeline& timeline, 
                                      ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

void adal::Solver::ExitPrimaryStep( const ada::AnalysisTimeline& timeline, 
                                    ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

void adal::Solver::EndAnalysisProcess( const ada::AnalysisTimeline& timeline, 
                                       ada::NumericalModel& model )
{
	// nothing to do in base implementation
}

bool adal::Solver::IsDebugEnabled( void ) const
{
	return _debug;
	// 	return _enableDebug;
}

void adal::Solver::ToggleDebug( bool state )
{
	_debug = state;
}

void adal::Solver::ToggleVerbosity( bool state )
{
	_verbose = state;
}

void adal::Solver::ToggleMute( bool state )
{
	_mute = state;
}

void adal::Solver::Run( ada::AnalysisTimeline& timeline, ada::NumericalModel& model )
{
	timeline.Reset();
	StartAnalysisProcess(timeline, model);
	bool enteredPrimaryStep = true;
	EnterPrimaryStep(timeline, model);
	do 
	{
		// if we have crossed a snapshot mark since last tick, open new
		// primary step
		if (timeline.HasCrossedSnapshotMark() && !enteredPrimaryStep)
		{
			DispatchMessage(adalm::SnapshotEndMessage(GetAnalysisInformation()));
			EnterPrimaryStep(timeline, model);
			enteredPrimaryStep = true;
		}

		EnterSecondaryStep(timeline, model);

		// calculate next clock tick
		_clock.CalculateNextTick(timeline, model);

		// execute secondary step
		ExecuteStep(timeline, model);

		// tick analysis clock
		timeline.Tick();

		ExitSecondaryStep(timeline, model);

		// if after advancing time we have crossed a snapshot mark,
		// close current primary step
		if (timeline.HasCrossedSnapshotMark())
		{
			DispatchMessage(adalm::SnapshotStartMessage(GetAnalysisInformation()));
			ExitPrimaryStep(timeline, model);
			enteredPrimaryStep = false;
		}
	} while (timeline.GetCurrentTimeMark() < timeline.EndTime());
	if (enteredPrimaryStep)
	{	// a primary step was left opened; close it
		DispatchMessage(adalm::SnapshotStartMessage(GetAnalysisInformation()));
		ExitPrimaryStep(timeline, model);
		DispatchMessage(adalm::SnapshotEndMessage(GetAnalysisInformation()));
	}

	EndAnalysisProcess(timeline, model);
}

void adal::Solver::AddTracingInformation( asmm::Message& message ) const
{
  message.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(1000, _T("Solver")));
}

bool adal::Solver::IsGPUCapable( void ) const
{
  return DoIsGPUCapable() && _clock.IsGPUCapable();
}

void adal::Solver::PrepareGPUData( ada::NumericalModel&, ada::AnalysisTimeline& )
{
  // nothing to do in base implementation
}

size_type adal::Solver::GetGPUThreadsRequired( const ada::NumericalModel& ) const
{
  return 10;
}

bool adal::Solver::IsCPUCapable( void ) const
{
  // Specialized implementation should explicitly indicate that is
  // NOT able to run in CPU
  return true;
}

void adal::Solver::AllocateGPUData( ada::NumericalModel&, ada::AnalysisTimeline& )
{
  // nothing to do in base implementation
}

bool adal::Solver::DoIsGPUCapable( void ) const
{
  // Specialized implementation should explicitly indicate that is
  // able to run in GPU
  return false;
}

void adal::Solver::StartAnalysisProcessOnGPU( const ada::AnalysisTimeline&, 
                                              afm::RelativePointer&,
                                              adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::EnterPrimaryStepOnGPU( const ada::AnalysisTimeline&, 
                                          afm::RelativePointer&,
                                          adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::EnterSecondaryStepOnGPU( const ada::AnalysisTimeline&, afm::RelativePointer&,
                                            adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::ExecuteStepOnGPU( const ada::AnalysisTimeline&, afm::RelativePointer&,
                                     adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::ExitSecondaryStepOnGPU( const ada::AnalysisTimeline&, afm::RelativePointer&,
                                           adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::ExitPrimaryStepOnGPU( const ada::AnalysisTimeline&, afm::RelativePointer&,
                                         adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::EndAnalysisProcessOnGPU( const ada::AnalysisTimeline&, afm::RelativePointer&,
                                            adal::ExternalSolverFacade&)
{
  // nothing to do in base implementation
}

void adal::Solver::RunOnGPU( ada::AnalysisTimeline& timeline, 
                             afm::RelativePointer& reducedModelPtr,
                             adal::ExternalSolverFacade& solverFacade)
{
  // init element bucket pointers
  ada::ReducedNumericalModel& model = 
    absref<ada::ReducedNumericalModel>(reducedModelPtr);
  model.GetOperator().InitElementBuckets();


  timeline.Reset();
  StartAnalysisProcessOnGPU(timeline, reducedModelPtr, solverFacade);
  bool enteredPrimaryStep = true;
  EnterPrimaryStepOnGPU(timeline, reducedModelPtr, solverFacade);
  do 
  {
    // if we have crossed a snapshot mark since last tick, open new
    // primary step
    if (timeline.HasCrossedSnapshotMark() && !enteredPrimaryStep)
    {
      solverFacade.DispatchMessageAsync(adalm::SnapshotEndMessage(GetAnalysisInformation()));
      solverFacade.EndResultCollectionRound();
      EnterPrimaryStepOnGPU(timeline, reducedModelPtr, solverFacade);
      enteredPrimaryStep = true;
    }

    EnterSecondaryStepOnGPU(timeline, reducedModelPtr, solverFacade);

    // calculate next clock tick
    _clock.CalculateNextTickOnGPU(timeline, reducedModelPtr, solverFacade);

    // execute secondary step
    ExecuteStepOnGPU(timeline, reducedModelPtr, solverFacade);

    // tick analysis clock
    timeline.Tick();

    ExitSecondaryStepOnGPU(timeline, reducedModelPtr, solverFacade);

    // if after advancing time we have crossed a snapshot mark,
    // close current primary step
    if (timeline.HasCrossedSnapshotMark())
    {
      solverFacade.StartResultCollectionRound(absref<ada::ReducedNumericalModel>(reducedModelPtr));
      solverFacade.DispatchMessageAsync(adalm::SnapshotStartMessage(GetAnalysisInformation()));
      ExitPrimaryStepOnGPU(timeline, reducedModelPtr, solverFacade);
      enteredPrimaryStep = false;
    }
  } while (timeline.GetCurrentTimeMark() < timeline.EndTime());
  if (enteredPrimaryStep)
  {	// a primary step was left opened; close it
    solverFacade.StartResultCollectionRound(absref<ada::ReducedNumericalModel>(reducedModelPtr));
    solverFacade.DispatchMessageAsync(adalm::SnapshotStartMessage(GetAnalysisInformation()));
    ExitPrimaryStepOnGPU(timeline, reducedModelPtr, solverFacade);
    solverFacade.DispatchMessageAsync(adalm::SnapshotEndMessage(GetAnalysisInformation()));
    solverFacade.EndResultCollectionRound();
  }
  else if (solverFacade.IsCollectionRoundActive())
  { // last collection round has not finished yet;
    solverFacade.DispatchMessageAsync(adalm::SnapshotEndMessage(GetAnalysisInformation()));
    solverFacade.EndResultCollectionRound();
  }
  EndAnalysisProcessOnGPU(timeline, reducedModelPtr, solverFacade);

  // Since message collection is asynchronous when using the GPU version of the
  // solver, in order to avoid missing messages (or other conditions caused
  // by detaching listeners while messages are still in transit), we wait for
  // all messages to be processed before continuing.
  solverFacade.FlushResultCollection();
}
