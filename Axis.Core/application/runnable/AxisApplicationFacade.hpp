#pragma once
#include "application/runnable/AxisApplication.hpp"
#include "application/agents/AnalysisAgent.hpp"
#include "application/agents/ParserAgent.hpp"
#include "application/agents/BootstrapAgent.hpp"
#include "application/agents/ConfigurationLoaderAgent.hpp"
#include "ConfigurationManagerImpl.hpp"

namespace axis { namespace application { namespace runnable {

class AxisApplicationFacade : public axis::application::runnable::AxisApplication
{
public:
	AxisApplicationFacade(void);
	virtual ~AxisApplicationFacade(void);
	virtual void Destroy( void ) const;
	virtual bool IsSystemReady(void) const;
	virtual bool IsJobRunning( void ) const;
	virtual axis::String GetApplicationName( void ) const;
	virtual int GetApplicationMajorVersion( void ) const;
	virtual int GetApplicationMinorVersion( void ) const;
	virtual int GetApplicationRevisionNumber( void ) const;
	virtual int GetApplicationBuildNumber( void ) const;
	virtual axis::String GetApplicationReleaseName( void ) const;
	virtual void Bootstrap( void );
	virtual ConfigurationManager& Configuration( void );
	virtual const ConfigurationManager& Configuration( void ) const;
	virtual void SubmitJob( const axis::application::jobs::JobRequest& job );
	virtual void RunCurrentJob( void );
	virtual const axis::application::jobs::StructuralAnalysis& GetJobWorkspace( void ) const;
	virtual const axis::services::management::GlobalProviderCatalog& GetModuleManager( void ) const;
	virtual axis::services::management::PluginLink GetPluginLinkInformation( size_type index ) const;
	virtual size_type GetPluginLinkCount( void ) const;
private:
  void BroadcastAppBanner(void);

	// this one encapsulates application configuration management
	// stuffs
	axis::application::runnable::ConfigurationManagerImpl _configManager;

	// our agents or subsystems if you like
	axis::application::agents::ConfigurationLoaderAgent   _configurationAgent;
	axis::application::agents::BootstrapAgent             _bootstrapAgent;
	axis::application::agents::ParserAgent                _parserAgent;
	axis::application::agents::AnalysisAgent              _analysisAgent;
				
	// the job for which we are responsible
	const axis::application::jobs::JobRequest *_currentJob;

	// status flags
	bool _initialized;
	bool _jobParsed;
	bool _jobRunning;
};

} } } // namespace axis::application::runnable
