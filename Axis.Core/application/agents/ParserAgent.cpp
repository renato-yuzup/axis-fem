#include "ParserAgent.hpp"
#include "ParserAgentImpl.hpp"
#include "ParserAgentDispatcher.hpp"
#include "services/io/FileSystem.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/LogMessage.hpp"
#include "foundation/IOException.hpp"
#include "services/locales/Locale.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "foundation/date_time/Timespan.hpp"

namespace aaa = axis::application::agents;
namespace aaj = axis::application::jobs;
namespace asi = axis::services::io;
namespace asmm = axis::services::messaging;
namespace asmg = axis::services::management;
namespace af = axis::foundation;
namespace afd = axis::foundation::date_time;
namespace afu = axis::foundation::uuids;

aaa::ParserAgent::ParserAgent( void )
{
	pimpl_ = NULL;
  dispatcher_ = new ParserAgentDispatcher(this);
}

aaa::ParserAgent::~ParserAgent( void )
{
  if (pimpl_ != NULL) delete pimpl_;
  delete dispatcher_;
}

void aaa::ParserAgent::AddPreProcessorSymbol( const axis::String& symbolName )
{
	if (pimpl_ == NULL)
	{
		throw af::InvalidOperationException(_T("Must set up agent first."));
	}
	pimpl_->AddPreProcessorSymbol(symbolName);
}

void aaa::ParserAgent::ClearPreProcessorSymbols( void )
{
	if (pimpl_ == NULL)
	{
		throw af::InvalidOperationException(_T("Must set up agent first."));
	}
	pimpl_->ClearPreProcessorSymbols();
}

void aaa::ParserAgent::SetUp( asmg::GlobalProviderCatalog& manager )
{
	if (pimpl_ != NULL)
	{
		delete pimpl_;
	}
	pimpl_ = new ParserAgentImpl(manager);
}

void aaa::ParserAgent::ReadAnalysis( const axis::String& masterFileName, 
														  const axis::String& baseIncludePath, 
														  const axis::String& outputLocationPath )
{
	if (pimpl_ == NULL)
	{
		throw af::InvalidOperationException(_T("Must set up agent first."));
	}

	// verify path existence
	if (!asi::FileSystem::ExistsFile(masterFileName))
	{
		DispatchMessage(asmm::ErrorMessage(0x300403, 
			_T("Master input file not found. Please check if file exists and you have sufficient permissions to read it."), 
			_T("Analysis file not found")));
		throw af::IOException();
	}
	if (!asi::FileSystem::IsDirectory(baseIncludePath))
	{
		DispatchMessage(asmm::ErrorMessage(0x300404, 
			_T("Include path not found. Check if it points to a directory and you have sufficient permissions to access it."), 
			_T("Include folder not found")));
		throw af::IOException();
	}

  pimpl_->BuildAnalysis(outputLocationPath);
  aaj::StructuralAnalysis& ws = pimpl_->GetAnalysis();
  ws.SetId(afu::Uuid::GenerateRandom());
  dispatcher_->LogParseStart(ws, masterFileName, baseIncludePath);

	pimpl_->ConnectListener(*dispatcher_);
	pimpl_->Parse(masterFileName, baseIncludePath);
	pimpl_->DisconnectListener(*dispatcher_);
}

aaj::StructuralAnalysis& aaa::ParserAgent::GetAnalysis( void ) const
{
	if (pimpl_ == NULL)
	{
		throw af::InvalidOperationException(_T("Must set up agent first."));
	}
	return pimpl_->GetAnalysis();
}

void aaa::ParserAgent::AddTracingInformation( asmm::Message& message ) const
{
  message.GetTraceInformation().AddTraceInfo(asmm::TraceInfo(500, _T("Parser")));
}
