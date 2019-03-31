#include "AnalysisAgent.hpp"
#include <iostream>
#include "foundation/NotSupportedException.hpp"

#include "application/jobs/FiniteElementAnalysis.hpp"
#include "application/jobs/WorkFolder.hpp"
#include "application/output/collectors/messages/AnalysisStartupMessage.hpp"
#include "application/output/collectors/messages/AnalysisEndMessage.hpp"
#include "application/output/collectors/messages/AnalysisStepStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisStepEndMessage.hpp"
#include "domain/algorithms/Solver.hpp"
#include "domain/algorithms/messages/SnapshotStartMessage.hpp"
#include "domain/algorithms/messages/SnapshotEndMessage.hpp"
#include "System.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/messaging/LogMessage.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "foundation/units/DigitalUnit.hpp"
#include "domain/analyses/AnalysisInfo.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"

namespace aaa = axis::application::agents;
namespace aaocm = axis::application::output::collectors::messages;
namespace aaj = axis::application::jobs;
namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace adalm = axis::domain::algorithms::messages;
namespace adbc = axis::domain::boundary_conditions;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace asl = axis::services::locales;
namespace asmm = axis::services::messaging;
namespace af = axis::foundation;
namespace afu = axis::foundation::units;
namespace asd = axis::services::diagnostics;
namespace asdi = axis::services::diagnostics::information;
namespace afdt = axis::foundation::date_time;


namespace {
  inline void BuildTransientAnalysisCollectionMessage(axis::String& msg1, axis::String& msg2, 
    const ada::TransientAnalysisInfo& info, bool isCollectionStarting)
  {
    uint64 iterationIndex = info.GetIterationIndex();
    real time = info.GetCurrentAnalysisTime();
    real lastDt = info.GetLastTimeStep();

    if (isCollectionStarting)
    {
      msg1 += _T("ITERATION ") + axis::String::int_parse(iterationIndex, 6) + _T(" --> ");
      msg1 += _T("BEGIN COLLECT PHASE");
      msg2 = _T("          ");
      msg2 += _T("time = ") + axis::String::double_parse(time).align_right(20);
      msg2 += _T(", last dt = ") + axis::String::double_parse(lastDt).align_right(20);
    }
    else
    {
      msg1 += _T("                 --> END COLLECT PHASE");
      msg2.clear();
    }
  }

  inline void BuildGenericAnalysisCollectionMessage(axis::String& msg,
    const ada::AnalysisInfo& info, bool isCollectionStarting)
  {
    msg += isCollectionStarting? _T("BEGIN") : _T("END");
    msg += _T(" COLLECT PHASE");
  }
}

aaa::AnalysisAgent::AnalysisAgent( void )
{
	analysis_ = NULL;
	_model = NULL;
}

aaa::AnalysisAgent::~AnalysisAgent( void )
{
	_model = NULL;
}

void aaa::AnalysisAgent::SetUp( aaj::StructuralAnalysis& analysis )
{
	analysis_ = &analysis;
	_model = &analysis_->GetNumericalModel();
}

void aaa::AnalysisAgent::Run( void )
{
	ada::NumericalModel& model = *_model;
	aaj::StructuralAnalysis& analysis = *analysis_;
  aaj::WorkFolder& workFolder = analysis.GetWorkFolder();
	
	// reset our statistics
	_errorCount = 0;
	_warningCount = 0;
	_lastErrorMsg = _T("");

	// prepare analysis
  analysis.SetAnalysisStartTime(afdt::Timestamp::GetUTCTime());

	// write log data about analysis being executed
	LogAnalysisInformation();

	// prepare to launch steps
	analysis.GetNumericalModel().ResetMesh();

	// flood a notification message that we began analysis; this
	// will make everyone prepare for analysis and open output
	// streams
	ProcessMessage(aaocm::AnalysisStartupMessage());

	// execute each step sequentially
	for (int stepIdx = 0; stepIdx <  analysis.GetStepCount(); stepIdx++)
	{
    analysis.SetCurrentStepIndex(stepIdx);
		_stepErrorCount = 0;
		_stepWarningCount = 0;

    aaj::AnalysisStep& step = analysis.GetCurrentStep();
    adal::Solver& solver = step.GetSolver();
    ada::AnalysisTimeline& timeline = step.GetTimeline();

    aaj::FiniteElementAnalysis fea(model, solver, timeline, step.GetResults(), workFolder);
    fea.SetJobInformation(analysis.GetTitle(), analysis.GetId(), analysis.GetCreationDate());
    fea.SetStepInformation(step.GetName(), stepIdx);
    fea.ConnectListener(*this);
    fea.AttachMonitor(analysis.GetHealthMonitor());
    fea.AttachPostProcessor(step.GetPostProcessor());
		LogStepInformation();
    _stepStartTime = afdt::Timestamp::GetLocalTime();
    ApplyCurrentStepBoundaryConditions();
    fea.StartAnalysis();
    fea.DisconnectListener(*this);
		LogAfterStepInformation();

		// clear boundary conditions so that the subsequent step can
		// assign its own
		ClearBoundaryConditionsFromModel();

    // erase temporary files created in this step
    analysis.GetWorkFolder().ClearTempFiles();
	}
  analysis.SetCurrentStepIndex(-1);
  analysis.SetAnalysisEndTime(afdt::Timestamp::GetUTCTime());

	// write run summary
	LogAnalysisSummary();
	
	ProcessMessage(aaocm::AnalysisEndMessage());
}

void aaa::AnalysisAgent::LogAnalysisInformation( void )
{
	const asl::Locale& loc = asl::LocaleLocator::GetLocator().GetGlobalLocale();

	_analysisBeginState = asd::Process::GetCurrentProcess();
	String title = analysis_->GetTitle();
	if (title.empty()) title = _T("<untitled>");

	DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionOpen, _T("===   A N A L Y S I S   P H A S E   ===")));
	DispatchMessage(asmm::LogMessage(_T("ANALYSIS TITLE                 : ") + title));
	DispatchMessage(asmm::LogMessage(_T("UNIQUE IDENTIFIER FOR THIS RUN : ") + analysis_->GetId().ToString()));
	DispatchMessage(asmm::LogMessage(_T("ANALYSIS START TIME            : ") + loc.GetDataTimeLocale().ToShortDateTimeMillisString(analysis_->GetAnalysisStartTime())));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
	DispatchMessage(asmm::LogMessage(_T("HARDWARE INFORMATION AND AVAILABILITY :")));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
  DispatchMessage(asmm::LogMessage(_T("PHYSICAL PROCESSORS INSTALLED         : ") + String::int_parse((long)System::Environment().GetLocalProcessorPackageCount())));
  DispatchMessage(asmm::LogMessage(_T("TOTAL NUMBER OF CORES AVAILABLE       : ") + String::int_parse((long)System::Environment().GetLocalProcessorCoreCount())));
  DispatchMessage(asmm::LogMessage(_T("TOTAL OF LOGICAL PROCESSORS AVAILABLE : ") + String::int_parse((long)System::Environment().GetLocalLogicalProcessorCount())));
  DispatchMessage(asmm::LogMessage(_T("LOGICAL PROCESSORS ALLOCATED FOR USE  : ") + String::int_parse((long)System::Environment().GetLocalLogicalProcessorAvailableCount()) + _T("  <-- AFTER APPLYING RESTRICTIONS SET BY OS AND THE USER")));
  DispatchMessage(asmm::LogMessage(_T("MAXIMUM NUMBER OF WORKER THREADS      : ") + String::int_parse((long)System::Environment().GetMaxLocalWorkerThreads())));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionClose, _T("")));
}

void aaa::AnalysisAgent::LogStepInformation( void )
{
	aaj::AnalysisStep& step = analysis_->GetCurrentStep();
	adal::Solver& solver = step.GetSolver();
	asdi::SolverCapabilities caps = solver.GetCapabilities();

	String stepTitle = step.GetName();
	if (stepTitle.empty()) stepTitle = _T("<unnamed>");

	DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
	DispatchMessage(asmm::LogMessage(_T("STARTING ANALYSIS STEP ") + String::int_parse((long)analysis_->GetCurrentStepIndex()) + _T(" : ") + stepTitle));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
	DispatchMessage(asmm::LogMessage(_T("RUNNING A ") + BuildAnalysisTypeString()));
	DispatchMessage(asmm::LogMessage(_T("ELLECTED SOLVER  : ") + caps.GetSolverName()));
	DispatchMessage(asmm::LogMessage(_T("                   ") + caps.GetDescription()));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
	DispatchMessage(asmm::LogMessage(_T("SOLVER OUTPUT:")));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
}

void aaa::AnalysisAgent::LogAfterStepInformation( void )
{
	String errorsStr = String::int_parse(_stepErrorCount);
	String warnStr = String::int_parse(_stepWarningCount);
	
	size_t x = (errorsStr.size() > warnStr.size()) ? errorsStr.size() : warnStr.size();
	errorsStr = String(x - errorsStr.size(), ' ') + errorsStr + _T(" ERROR");
	warnStr = String(x - warnStr.size(), ' ') + warnStr + _T(" WARNING");

	if (_stepErrorCount != 1) errorsStr += _T("S");
	if (_stepWarningCount != 1) warnStr += _T("S");

  DispatchMessage(asmm::LogMessage(_T("<END OF SOLVER OUTPUT>")));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  DispatchMessage(asmm::LogMessage(_T("ANALYSIS STEP ENDED WITH :")));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
  DispatchMessage(asmm::LogMessage(errorsStr));
  DispatchMessage(asmm::LogMessage(warnStr));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
}

void aaa::AnalysisAgent::LogAnalysisSummary( void )
{
	const asl::Locale& loc = asl::LocaleLocator::GetLocator().GetGlobalLocale();

  afdt::Timestamp analysisStartTime = analysis_->GetAnalysisStartTime();
  afdt::Timestamp analysisEndTime = analysis_->GetAnalysisEndTime();
  afdt::Timespan analysisDuration = analysisEndTime - analysisStartTime;

  // calculate user and kernel times
	_analysisEndState = asd::Process::GetCurrentProcess();
	afdt::Timespan userTime = _analysisEndState.GetUserTime() - _analysisBeginState.GetUserTime();
	afdt::Timespan kernelTime = _analysisEndState.GetKernelTime() - _analysisBeginState.GetKernelTime();

	// calculate peak memory usage
	uint64 peakPhysMemSize = _analysisEndState.GetPeakPhysicalMemoryAllocated() - _analysisBeginState.GetPhysicalMemoryAllocated();
	uint64 peakVirtualMemSize = _analysisEndState.GetPeakVirtualMemoryAllocated() - _analysisBeginState.GetVirtualMemoryAllocated();

	// print results
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionClose, _T("")));	// close analysis phase section
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionOpen, _T("===   A N A L Y S I S   S U M M A R Y   ===")));

	if (_errorCount > 0)
	{
		DispatchMessage(asmm::LogMessage(_T("ANALYSIS COMPLETED WITH ERRORS. RESULTS MIGHT NOT HAVE BEEN WRITTEN OR ARE INVALID.")));
	}
	else
	{
		DispatchMessage(asmm::LogMessage(_T("ANALYSIS COMPLETED SUCCESSFULLY. CHECK OUTPUT FILES FOR RESULTS.")));
	}
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
	DispatchMessage(asmm::LogMessage(String::int_parse(_errorCount).align_right(5) + _T(" ERROR") + ((_errorCount != 1)? _T("S") : _T(""))));
	DispatchMessage(asmm::LogMessage(String::int_parse(_warningCount).align_right(5) + _T(" WARNING") + ((_warningCount != 1)? _T("S") : _T(""))));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
	DispatchMessage(asmm::LogMessage(_T("ANALYSIS ENDED ON : ") + loc.GetDataTimeLocale().ToShortDateTimeMillisString(analysisEndTime)));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
	DispatchMessage(asmm::LogMessage(_T("ANALYSIS TIMINGS :")));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
	DispatchMessage(asmm::LogMessage(_T("USER (PROGRAM) TIME  : ") + loc.GetDataTimeLocale().ToShortTimeRangeString(userTime)));
	DispatchMessage(asmm::LogMessage(_T("KERNEL (SYSTEM) TIME : ") + loc.GetDataTimeLocale().ToShortTimeRangeString(kernelTime)));
	DispatchMessage(asmm::LogMessage(_T("WALL CLOCK TIME      : ") + loc.GetDataTimeLocale().ToShortTimeRangeString(analysisDuration)));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
	DispatchMessage(asmm::LogMessage(_T("PEAK MEMORY USAGE (ANALYSIS PHASE ONLY) :")));
	DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
	DispatchMessage(asmm::LogMessage(_T("MAX PHYSICAL MEM. USED : ") + 
    String::int_parse((long)afu::DigitalUnit::Convert(peakPhysMemSize,
    afu::DigitalUnit::FromScale(afu::DigitalUnit::ByteUnit()), 
    afu::DigitalUnit::ToScale(afu::DigitalUnit::MegaByteUnit()))) + _T(" MB")));
	DispatchMessage(asmm::LogMessage(_T("MAX VIRTUAL MEM. USED  : ") + String::int_parse(
    (long)afu::DigitalUnit::Convert(peakVirtualMemSize, 
    afu::DigitalUnit::FromScale(afu::DigitalUnit::ByteUnit()), 
    afu::DigitalUnit::ToScale(afu::DigitalUnit::MegaByteUnit()))) + _T(" MB")));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionClose, _T("*** END OF ANALYSIS")));
  DispatchMessage(asmm::LogMessage(_T("")));	// blank line
  DispatchMessage(asmm::LogMessage(_T("")));	// blank line
}

axis::String aaa::AnalysisAgent::BuildAnalysisTypeString( void ) const
{
	String s;
  aaj::AnalysisStep& curStep = analysis_->GetCurrentStep();
	adal::Solver& solver = curStep.GetSolver();
	asdi::SolverCapabilities caps = solver.GetCapabilities();

	int i = 0;
	i += caps.DoesAccountMaterialNonlinearity()? 1 : 0;
	i += caps.DoesAccountGeometricNonlinearity()? 1 : 0;
	i += caps.DoesAccountBoundaryConditionsNonlinearity()? 1 : 0;

	// is it linear?
	if (i == 0)		// no non-linearity
	{
		s = _T("LINEAR ");
	}
	else
	{
		s = _T("NONLINEAR ");
	}

	// is it static?
	if (!caps.DoesSolveTimeDependentProblems())
	{
		s += _T("STATIC ANALYSIS");
	}
	else
	{
		s += _T("DYNAMIC ANALYSIS");
	}

	// if it is nonlinear, add more details
	if (i > 0) 
	{
		s += _T(" WITH ");
		if (caps.DoesAccountMaterialNonlinearity())
		{	
			i--;
			s += _T("NONLINEAR MATERIALS");
		}
		if (caps.DoesAccountGeometricNonlinearity())
		{
			i--;
			s += (i == 0)? _T(" AND ") : _T(", ");
			s += _T("FINITE DEFORMATIONS");
		}
		if (caps.DoesAccountBoundaryConditionsNonlinearity())
		{
			i--;
			s += _T(" AND CONTACT");
		}
	}

	return s;
}

void aaa::AnalysisAgent::ProcessEventMessageLocally( const asmm::EventMessage& volatileMessage )
{
	// ignore if the message has no trace information
	if (volatileMessage.GetTraceInformation().Empty()) return;

	const asmm::TraceInfoCollection::value_type& msgInfo = 
    volatileMessage.GetTraceInformation().PeekInfo();
  aaj::AnalysisStep& curStep = analysis_->GetCurrentStep();
	const adal::Solver& solver = curStep.GetSolver();
	
	// check if the last source tag came from the solver we are
	// using
	if (msgInfo.SourceId() == solver.GetSolverEventSourceId())
	{	// yes it came; count possible errors and warnings
		if (volatileMessage.IsError())
		{
			_errorCount++;
			_lastErrorMsg = volatileMessage.GetDescription();
		}
		else if (volatileMessage.IsWarning())
		{
			_warningCount++;
		}
	}
}

void aaa::AnalysisAgent::ProcessResultMessageLocally(const asmm::ResultMessage& volatileMessage)
{
  const asl::Locale& loc = asl::LocaleLocator::GetLocator().GetGlobalLocale();
  auto& timeLoc = loc.GetDataTimeLocale();
  auto& now = afdt::Timestamp::GetLocalTime();
  auto& elapsedTime = now - _stepStartTime;

  String consoleMsg1, consoleMsg2;
  String percentCompletedStr(8, ' '); //(59, ' ');
  consoleMsg1.reserve(300);
  consoleMsg2.reserve(300);
  consoleMsg1 = _T("   [") + timeLoc.ToShortDateTimeMillisString(now) + _T("] ::: ");
 
  if (adalm::SnapshotStartMessage::IsOfKind(volatileMessage))
  {
    auto& msg = static_cast<const adalm::SnapshotStartMessage&>(volatileMessage);
    auto& analysisInfo = msg.GetAnalysisInformation();
    if (analysisInfo.GetAnalysisType() == ada::AnalysisInfo::TransientAnalysis)
    {
      auto& info = static_cast<const ada::TransientAnalysisInfo&>(analysisInfo);
      BuildTransientAnalysisCollectionMessage(consoleMsg1, consoleMsg2, info, true);
      std::wcout << consoleMsg1.c_str() << std::endl;
      std::wcout << consoleMsg2.c_str();
    }
    else
    {
      BuildGenericAnalysisCollectionMessage(consoleMsg1, analysisInfo, true);
      std::wcout << consoleMsg1.c_str();
    }
    std::wcout << std::endl;
  }
  else if (adalm::SnapshotEndMessage::IsOfKind(volatileMessage))
  {
    auto& msg = static_cast<const adalm::SnapshotEndMessage&>(volatileMessage);
    auto& analysisInfo = msg.GetAnalysisInformation();
    if (analysisInfo.GetAnalysisType() == ada::AnalysisInfo::TransientAnalysis)
    {
      auto& info = static_cast<const ada::TransientAnalysisInfo&>(analysisInfo);
      BuildTransientAnalysisCollectionMessage(consoleMsg1, consoleMsg2, info, false);
      real fractionDone = (info.GetCurrentAnalysisTime() - info.GetStartTime()) / 
        (info.GetEndTime() - info.GetStartTime());
      int percentDone = (int)(fractionDone * 10000);
      percentCompletedStr += String::int_parse(percentDone / 100) + _T(".") +
        String::int_parse(percentDone % 100) + _T("% done  ---  Remaining time: ");
      auto remainingTime = elapsedTime * ((1 - fractionDone) / fractionDone);
      if (remainingTime.HasWholeDays())
      {
        percentCompletedStr += String::int_parse(remainingTime.GetDays()) + _T(" days, ");
      }
      if (remainingTime.GetHours() > 0)
      {
        percentCompletedStr += String::int_parse(remainingTime.GetHours()) + _T("h ");
      }
      percentCompletedStr += String::int_parse(remainingTime.GetMinutes()) + _T("min, ");
      percentCompletedStr += String::int_parse(remainingTime.GetSeconds()) + _T("s");
      percentCompletedStr += _T(" (estimated)");
      std::wcout << consoleMsg1.c_str() << std::endl;
      std::wcout << percentCompletedStr.c_str();
    }
    else
    {
      BuildGenericAnalysisCollectionMessage(consoleMsg1, analysisInfo, false);
      std::wcout << consoleMsg1.c_str();
    }
    std::wcout << std::endl << std::endl;
  }
}

void aaa::AnalysisAgent::ApplyCurrentStepBoundaryConditions( void )
{
	ada::NumericalModel& model = analysis_->GetNumericalModel();
	aaj::AnalysisStep& step = analysis_->GetCurrentStep();
  adc::DofList& allBcList = model.AllBoundaryConditions();
	ApplyBoundaryConditionSetToModel(model.NodalLoads(), allBcList, step.NodalLoads());
	ApplyBoundaryConditionSetToModel(model.AppliedDisplacements(), allBcList, step.Displacements());
	ApplyBoundaryConditionSetToModel(model.AppliedAccelerations(), allBcList, step.Accelerations());
	ApplyBoundaryConditionSetToModel(model.AppliedVelocities(), allBcList, step.Velocities());
	ApplyBoundaryConditionSetToModel(model.Locks(), allBcList, step.Locks());
}

void aaa::AnalysisAgent::ApplyBoundaryConditionSetToModel( 
  adc::DofList& bcTypeList, adc::DofList& generalList, adc::BoundaryConditionCollection& bcList )
{
	size_type count = bcList.Count();
	#pragma omp sections
	{
		#pragma omp section
		{
			#pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
			for (size_type i = 0; i < count; ++i)
			{
				ade::DoF& dof = bcList.GetKey(i);
				adbc::BoundaryCondition& bc = bcList.Get(i);
				dof.SetBoundaryCondition(bc);
				bc.SetDoF(&dof);
			}
		}
		#pragma omp section
		{
			for (size_type i = 0; i < count; ++i)
			{
				ade::DoF& dof = bcList.GetKey(i);
				bcTypeList.Add(dof);
        generalList.Add(dof);
			}
		}
	}
}

void aaa::AnalysisAgent::ClearBoundaryConditionsFromModel( void )
{
	ada::NumericalModel& model = analysis_->GetNumericalModel();
	ClearBoundaryConditionSet(model.NodalLoads());
	ClearBoundaryConditionSet(model.Locks());
	ClearBoundaryConditionSet(model.AppliedDisplacements());
	ClearBoundaryConditionSet(model.AppliedAccelerations());
	ClearBoundaryConditionSet(model.AppliedVelocities());
  model.AllBoundaryConditions().Clear();
}

void aaa::AnalysisAgent::ClearBoundaryConditionSet( adc::DofList& dofList )
{
	size_type count = dofList.Count();
	#pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
	for (size_type i = 0; i < count; ++i)
	{
		ade::DoF& dof = dofList[i];
		adbc::BoundaryCondition& bc = dof.GetBoundaryCondition();
		dof.RemoveBoundaryCondition();
		bc.SetDoF(NULL);
	}
  dofList.Clear();
}
