#include "ParserAgentDispatcher.hpp"
#include "ParserAgentImpl.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/jobs/WorkFolder.hpp"
#include "application/output/ChainMetadata.hpp"
#include "application/output/ResultBucket.hpp"
#include "services/locales/Locale.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/LogMessage.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "foundation/AnalysisReadException.hpp"

namespace aaag = axis::application::agents;
namespace aaj = axis::application::jobs;
namespace aao = axis::application::output;
namespace aaam = axis::application::agents::messages;
namespace asi = axis::services::io;
namespace asmm = axis::services::messaging;
namespace aaam = axis::application::agents::messages;
namespace asll = axis::services::locales;
namespace aaj = axis::application::jobs;
namespace afd = axis::foundation::date_time;

aaag::ParserAgent::ParserAgentDispatcher::ParserAgentDispatcher( ParserAgent *parent )
{
  parent_ = parent;
}

aaag::ParserAgent::ParserAgentDispatcher::~ParserAgentDispatcher( void )
{
  // nothing to do here
}

void aaag::ParserAgent::ParserAgentDispatcher::DoProcessResultMessage( 
  asmm::ResultMessage& volatileMessage )
{
  if (aaam::ParseFinishMessage::IsOfKind(volatileMessage))
  {
    LogParseEndSummary(static_cast<aaam::ParseFinishMessage&>(volatileMessage));
  }
  else if (aaam::ParseRoundEndMessage::IsOfKind(volatileMessage))
  {
    LogParsePartialSummary(static_cast<aaam::ParseRoundEndMessage&>(volatileMessage));
  }
}

void aaag::ParserAgent::ParserAgentDispatcher::LogParseEndSummary( 
  const aaam::ParseFinishMessage& message )
{
  const aaj::StructuralAnalysis& ws = message.GetAnalysis();
  const asll::Locale& loc = asll::LocaleLocator::GetLocator().GetGlobalLocale();
  axis::String symbolList = parent_->pimpl_->GetPreProcessorSymbolList();
  afd::Timestamp parseEndTime = afd::Timestamp::GetUTCTime();
  afd::Timespan parseElapsedTime = parseEndTime - ws.GetCreationDate();

  // put notice if errors occurred
  if (message.GetErrorCount() > 0)
  {
    parent_->DispatchMessage(asmm::LogMessage(_T("NOTICE :   ERRORS OCCURRED WHEN TRYING TO LOAD ANALYSIS. PARTIAL RESULTS ARE PRESENTED BELOW.")));
    parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
    parent_->DispatchMessage(asmm::LogMessage(_T("PARSING PARTIAL RESULTS :")));
  }
  else
  {
    parent_->DispatchMessage(asmm::LogMessage(_T("Model parsing operation completed with no errors.")));
    parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
    parent_->DispatchMessage(asmm::LogMessage(_T("PARSING FINAL RESULTS :")));
  }

  // print parsing general information
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
  parent_->DispatchMessage(asmm::LogMessage(_T("FINISHED AT  : ") + loc.GetDataTimeLocale().ToShortDateTimeMillisString(parseEndTime)));
  parent_->DispatchMessage(asmm::LogMessage(_T("ELAPSED TIME : ") + loc.GetDataTimeLocale().ToShortTimeRangeString(parseElapsedTime)));
  parent_->DispatchMessage(asmm::LogMessage(_T("FLAGS USED   : ") + symbolList));
  parent_->DispatchMessage(asmm::LogMessage(_T("ERRORS       : ") + axis::String::int_parse(message.GetErrorCount())));
  parent_->DispatchMessage(asmm::LogMessage(_T("WARNINGS     : ") + axis::String::int_parse(message.GetWarningCount())));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));

  if (message.GetErrorCount() > 0)
  {
    parent_->DispatchMessage(asmm::WarningMessage(0x200401, _T("Model read finished with errors. Program execution will be halted.")));
    throw axis::foundation::AnalysisReadException(_T("Failed to load analysis."));
  }

  // print model summary
  PrintModelSummary(ws);

  // print output file list
  PrintOutputList(ws);

  axis::String s(_T("Exiting parse operation after "));
  s += axis::String::int_parse((long)message.GetReadRoundCount());
  s += _T(" round");
  s += (message.GetReadRoundCount() == 1? _T(".") : _T("s."));
  parent_->DispatchMessage(asmm::LogMessage(s));
}

void aaag::ParserAgent::ParserAgentDispatcher::LogParseStart( const aaj::StructuralAnalysis& analysis, const axis::String& masterInputFile, const axis::String& baseIncludePath )
{
  afd::Timestamp submissionDate = analysis.GetCreationDate();
  const asll::Locale& loc = asll::LocaleLocator::GetLocator().GetDefaultLocale();
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionOpen, _T("===   J O B   I N F O R M A T I O N   ===")));
  parent_->DispatchMessage(asmm::LogMessage(_T("JOB SUBMITTED ON   : ") + loc.GetDataTimeLocale().ToShortDateTimeMillisString(submissionDate)));
  parent_->DispatchMessage(asmm::LogMessage(_T("MASTER INPUT FILE  : ") + masterInputFile));
  parent_->DispatchMessage(asmm::LogMessage(_T("BASE INCLUDE FILE  : ") + baseIncludePath));
  parent_->DispatchMessage(asmm::LogMessage(_T("BASE OUTPUT FOLDER : ") + analysis.GetWorkFolder().GetLocation()));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionClose, _T("")));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::SectionOpen, _T("===   M O D E L   P A R S I N G   P H A S E   ===")));
  parent_->DispatchMessage(asmm::LogMessage(_T("Model parsing operation began on ") + loc.GetDataTimeLocale().ToShortDateTimeMillisString(submissionDate)));
}

void aaag::ParserAgent::ParserAgentDispatcher::LogParsePartialSummary( const aaam::ParseRoundEndMessage& message )
{
  axis::String s(_T("For debugging purposes: ROUND = "));
  s += axis::String::int_parse(message.GetRoundIndex());
  s += _T(", UNDEFINED_SYMBOLS = ");
  s += axis::String::int_parse(message.GetUndefinedSymbolCount());
  s += _T(", DEFINED_SYMBOLS = ");
  s += axis::String::int_parse(message.GetDefinedSymbolCount());
  s += _T(", ERRORS_SO_FAR = ");
  s += axis::String::int_parse(message.GetErrorCount());
  s += _T(", WARNINGS_SO_FAR = ");
  s += axis::String::int_parse(message.GetWarningCount());
  parent_->DispatchMessage(asmm::InfoMessage(0, s, asmm::InfoMessage::InfoDebugLevel));
}

void aaag::ParserAgent::ParserAgentDispatcher::PrintModelSummary( 
  const aaj::StructuralAnalysis& analysis )
{
	axis::String totalNodes = axis::String::int_parse(analysis.GetNumericalModel().Nodes().Count());
	axis::String totalElements = axis::String::int_parse(analysis.GetNumericalModel().Elements().Count());
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
  parent_->DispatchMessage(asmm::LogMessage(_T("MODEL SUMMARY :")));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
  parent_->DispatchMessage(asmm::LogMessage(_T("NODES    : ") + totalNodes));
  parent_->DispatchMessage(asmm::LogMessage(_T("ELEMENTS : ") + totalElements));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
}

void aaag::ParserAgent::ParserAgentDispatcher::PrintOutputList(const aaj::StructuralAnalysis& analysis)
{
	size_type stepCount = analysis.GetStepCount();

	parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockOpen));
  parent_->DispatchMessage(asmm::LogMessage(_T("OUTPUT COLLECTION INFORMATION :")));
  parent_->DispatchMessage(asmm::LogMessage(_T("** CHAIN OUTPUT PATH IS RELATIVE TO THE JOB BASE OUTPUT FOLDER **")));
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
  for (size_type stepIdx = 0; stepIdx < stepCount; ++stepIdx)
	{
		const aaj::AnalysisStep& step = analysis.GetStep(stepIdx);
		String stepName = step.GetName().empty() ? _T("<unnamed>") : step.GetName();
    parent_->DispatchMessage(asmm::LogMessage(_T("COLLECTOR CHAINS IN ANALYSIS STEP ") + String::int_parse(stepIdx+1) + _T(" : ") +  stepName));
    parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));

    const aao::ResultBucket& bucket = step.GetResults();
		size_type chainCount = bucket.GetChainCount();
		for (size_type chainIdx = 0; chainIdx < chainCount; ++chainIdx)
		{
      aao::ChainMetadata metadata = bucket.GetChainMetadata(chainIdx);
      parent_->DispatchMessage(asmm::LogMessage(_T("CHAIN ") + axis::String::int_parse((long)chainIdx+1, 2) + _T(":: WILL OUTPUT TO '") + metadata.GetOutputFileName() + _T("'")));
      parent_->DispatchMessage(asmm::LogMessage(_T("           FORMAT IS IDENTIFIED AS A ") + metadata.GetTitle()));
      parent_->DispatchMessage(asmm::LogMessage(_T("           *** ") + metadata.GetShortDescription() + _T(" ***")));

      int collectorCount = metadata.GetCollectorCount();
			for (int collectorIdx = 0; collectorIdx < collectorCount; ++collectorIdx)
			{
				axis::String s = collectorIdx == 0? _T("           Data that will be stored here: - ") : 
                                            _T("                                          - ");
				s += metadata[collectorIdx];
				parent_->DispatchMessage(asmm::LogMessage(s));
			}		
		}
    parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
	}
  parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
	parent_->DispatchMessage(asmm::LogMessage(asmm::LogMessage::BlockClose));
}
