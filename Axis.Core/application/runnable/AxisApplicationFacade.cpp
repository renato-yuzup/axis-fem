#include "AxisApplicationFacade.hpp"
#include "services/messaging/LogMessage.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/ErrorMessage.hpp"

namespace aaj = axis::application::jobs;
namespace aar = axis::application::runnable;
namespace asmm = axis::services::messaging;
namespace asmg = axis::services::management;
namespace af = axis::foundation;
namespace afu = axis::foundation::uuids;

aar::AxisApplicationFacade::AxisApplicationFacade( void )
{
	// connect to agents
	_analysisAgent.ConnectListener(*this);
	_parserAgent.ConnectListener(*this);
	_bootstrapAgent.ConnectListener(*this);

	_initialized = false;
	_currentJob = NULL;
	_jobParsed = false;
	_jobRunning = false;
}

aar::AxisApplicationFacade::~AxisApplicationFacade( void )
{
	if (_currentJob != NULL)
	{
		_currentJob->Destroy();
	}
	_currentJob = NULL;
}

void aar::AxisApplicationFacade::Destroy( void ) const
{
	delete this;
}

bool aar::AxisApplicationFacade::IsSystemReady( void ) const
{
	return _initialized;
}

bool aar::AxisApplicationFacade::IsJobRunning( void ) const
{
	return _jobRunning;
}

void aar::AxisApplicationFacade::BroadcastAppBanner( void )
{
	int bannerWidth = 120;
  String rule(bannerWidth, '*');
  String versionStr = _T("version ") + String::int_parse((long)GetApplicationMajorVersion()) + _T(".") + 
						String::int_parse((long)GetApplicationMinorVersion()) + _T(".") + 
						String::int_parse((long)GetApplicationRevisionNumber()) + _T(" build ") + 
						String::int_parse((long)GetApplicationBuildNumber()) + _T(" \"") + 
						GetApplicationReleaseName() + _T("\"");
	String appName = GetApplicationName();
	appName.to_upper_case();

  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BannerStart));
  DispatchMessage(asmm::LogMessage(rule));
  DispatchMessage(asmm::LogMessage(_T("")));	// a blank line
  DispatchMessage(asmm::LogMessage(appName.align_center(bannerWidth)));
  DispatchMessage(asmm::LogMessage(_T("")));	// a blank line
  DispatchMessage(asmm::LogMessage(versionStr.align_right(bannerWidth)));
  DispatchMessage(asmm::LogMessage(_T("")));	// a blank line
  DispatchMessage(asmm::LogMessage(rule));
  DispatchMessage(asmm::LogMessage(_T("")));	// a blank line
  DispatchMessage(asmm::LogMessage(_T("")));	// a blank line
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::BannerEnd));
}

axis::String aar::AxisApplicationFacade::GetApplicationName( void ) const
{
	return _T("Axis Finite Element Analysis Program for Structural Mechanics");
}

int aar::AxisApplicationFacade::GetApplicationMajorVersion( void ) const
{
	return 0;
}

int aar::AxisApplicationFacade::GetApplicationMinorVersion( void ) const
{
	return 6;
}

int aar::AxisApplicationFacade::GetApplicationRevisionNumber( void ) const
{
	return 4;
}

int aar::AxisApplicationFacade::GetApplicationBuildNumber( void ) const
{
	return 670;	// last check-in version of axis.Solver.vcxproj
}

axis::String aar::AxisApplicationFacade::GetApplicationReleaseName( void ) const
{
	return _T("Echeveria");
}

void aar::AxisApplicationFacade::Bootstrap( void )
{
	_configurationAgent.SetConfigurationLocation(_configManager.GetConfigurationScriptPath());
	_configurationAgent.LoadConfiguration();
	BroadcastAppBanner();
	_bootstrapAgent.SetUp(_configurationAgent.GetConfiguration());
	_bootstrapAgent.Run();
	_initialized = true;
}

aar::ConfigurationManager& aar::AxisApplicationFacade::Configuration( void )
{
	return _configManager;
}

const aar::ConfigurationManager& aar::AxisApplicationFacade::Configuration( void ) const
{
	return _configManager;
}

void aar::AxisApplicationFacade::SubmitJob( const aaj::JobRequest& job )
{
	if (!IsSystemReady())
	{
		throw af::InvalidOperationException(_T("System not ready."));
	}
	if (_jobRunning)
	{
		throw af::InvalidOperationException(_T("Cannot submit a new job until the current one terminates."));
  }
  DispatchMessage(asmm::InfoMessage(0, 
                                _T("A new job has been submitted. Going to read master input file '") + 
                                job.GetMasterInputFilePath() + _T("' on '") + job.GetBaseIncludePath() + 
                                _T("'"), asmm::InfoMessage::InfoNormal));
	if (_currentJob != NULL)
	{
		_parserAgent.GetAnalysis().Destroy();
		_currentJob->Destroy();
	}
	_jobParsed = false;
	_currentJob = &job.Clone();
	
	// parse job
	_parserAgent.SetUp(_bootstrapAgent.GetModuleManager());
	_parserAgent.ClearPreProcessorSymbols();
	size_type flagCount = job.GetConditionalFlagsCount();
	for (size_type i = 0; i < flagCount; ++i)
	{
		_parserAgent.AddPreProcessorSymbol(job.GetConditionalFlag(i));
	}

  String inputFile = job.GetMasterInputFilePath();
  String includePath = job.GetBaseIncludePath();
  String outputPath = job.GetOutputFolderPath();
	_parserAgent.ReadAnalysis(inputFile, includePath, outputPath);
	_jobParsed = true;
  DispatchMessage(asmm::InfoMessage(0, 
                                _T("Job read successfully. Waiting for command to start analysis."), 
                                asmm::InfoMessage::InfoNormal));
}

void aar::AxisApplicationFacade::RunCurrentJob( void )
{
	if (!IsSystemReady())
	{
		throw af::InvalidOperationException(_T("System not ready."));
	}
	if (_currentJob == NULL || !_jobParsed)
	{
		throw af::InvalidOperationException(_T("There is no valid job to run."));
	}
	if (_jobRunning)
	{
		throw af::InvalidOperationException(_T("Job is already running."));
	}

  aaj::StructuralAnalysis& analysis = _parserAgent.GetAnalysis();
  afu::Uuid jobId = analysis.GetId();

  DispatchMessage(asmm::InfoMessage(0, _T("Start analysis command received. Going to run job ID=") + 
                                    jobId.ToString(), asmm::InfoMessage::InfoNormal));
// 	try
// 	{
		_jobRunning = false;
		_analysisAgent.SetUp(analysis);
		_analysisAgent.Run();
// 	}
// 	catch (af::AxisException& e)
// 	{
//     DispatchMessage(asmm::ErrorMessage(0, _T("Job ID=") + jobId.ToString() + 
//                                        _T(" failed with a critical error."), e, 
//                                        asmm::ErrorMessage::ErrorCritical));
// 		_jobRunning = false;
// 		throw;
// 	}
//   catch (...)
//   {
//     DispatchMessage(asmm::ErrorMessage(0, _T("Job ID=") + jobId.ToString() + 
//                         _T(" failed with a critical error. Exception cause could not be determined."), 
//                         asmm::ErrorMessage::ErrorCritical));
//     _jobRunning = false;
//     throw;
//   }
  DispatchMessage(asmm::InfoMessage(0, _T("Job ID=") + jobId.ToString() + _T(" terminated gracefully."), 
                                    asmm::InfoMessage::InfoNormal));
	_jobRunning = false;
}

const aaj::StructuralAnalysis& aar::AxisApplicationFacade::GetJobWorkspace( void ) const
{
	if (!(IsSystemReady() && _jobParsed))
	{
		throw af::InvalidOperationException(_T("There is no valid job submitted."));
	}
	return _parserAgent.GetAnalysis();
}

const asmg::GlobalProviderCatalog& aar::AxisApplicationFacade::GetModuleManager( void ) const
{
	if (!IsSystemReady())
	{
		throw af::InvalidOperationException(_T("System is not ready."));
	}
	return _bootstrapAgent.GetModuleManager();
}

asmg::PluginLink aar::AxisApplicationFacade::GetPluginLinkInformation( size_type index ) const
{
	return _bootstrapAgent.GetPluginLinkInfo(index);
}

size_type aar::AxisApplicationFacade::GetPluginLinkCount( void ) const
{
	return _bootstrapAgent.GetPluginLinkCount();
}
