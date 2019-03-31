#pragma once
#include "foundation/Axis.Core.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "application/runnable/ConfigurationManager.hpp"
#include "application/jobs/JobRequest.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/management/PluginLink.hpp"

namespace axis { namespace application { namespace runnable {

/// <summary>
/// Provides a single interface for access to all application features by external services.
/// </summary>
class AXISCORE_API AxisApplication : public axis::services::messaging::CollectorHub
{
public:
	static AxisApplication& CreateApplication(void);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	virtual ~AxisApplication(void);

	/**************************************************************************************************
		* <summary>	Destroys this object. </summary>
		**************************************************************************************************/
	virtual void Destroy(void) const = 0;

	/**************************************************************************************************
		* <summary>	Starts up application and load external plugins modules. </summary>
		**************************************************************************************************/
	virtual void Bootstrap(void) = 0;

	/**************************************************************************************************
		* <summary>	Returns the application configuration subsystem. </summary>
		*
		* <returns>	The configuration subsystem. </returns>
		**************************************************************************************************/
	virtual ConfigurationManager& Configuration(void) = 0;

	/**************************************************************************************************
		* <summary>	Returns the application configuration subsystem. </summary>
		*
		* <returns>	The configuration subsystem. </returns>
		**************************************************************************************************/
	virtual const ConfigurationManager& Configuration(void) const = 0;

	/**************************************************************************************************
		* <summary>	Places an analysis job to be executed. Previous job that has not run is replaced.
		* 				</summary>
		*
		* <param name="job">	The job information. </param>
		**************************************************************************************************/
	virtual void SubmitJob(const axis::application::jobs::JobRequest& job) = 0;

	/**************************************************************************************************
		* <summary>	Executes the current analysis job. </summary>
		**************************************************************************************************/
	virtual void RunCurrentJob(void) = 0;

 	/**************************************************************************************************
 		* <summary>	Queries if this application is ready to accept new jobs. </summary>
 		*
 		* <returns>	true if system ready, false if not. </returns>
 		**************************************************************************************************/
 	virtual bool IsSystemReady(void) const = 0;

	/**************************************************************************************************
		* <summary>	Queries if there is a job running. </summary>
		*
		* <returns>	true if a job is running, false otherwise. </returns>
		**************************************************************************************************/
	virtual bool IsJobRunning(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the application name. </summary>
		*
		* <returns>	The application name. </returns>
		**************************************************************************************************/
	virtual axis::String GetApplicationName(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the application major version. </summary>
		*
		* <returns>	The application major version. </returns>
		**************************************************************************************************/
	virtual int GetApplicationMajorVersion(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the application minor version. </summary>
		*
		* <returns>	The application minor version. </returns>
		**************************************************************************************************/
	virtual int GetApplicationMinorVersion(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the application revision number. </summary>
		*
		* <returns>	The application revision number. </returns>
		**************************************************************************************************/
	virtual int GetApplicationRevisionNumber(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the application build number. </summary>
		*
		* <returns>	The application build number. </returns>
		**************************************************************************************************/
	virtual int GetApplicationBuildNumber(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the application release name. </summary>
		*
		* <returns>	The application release name. </returns>
		**************************************************************************************************/
	virtual axis::String GetApplicationReleaseName(void) const = 0;

	/**************************************************************************************************
		* <summary>	Return the corresponding analysis for the current job already parsed. </summary>
		*
		* <returns>	The job analysis. </returns>
		**************************************************************************************************/
	virtual const axis::application::jobs::StructuralAnalysis& GetJobWorkspace(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns the program module manager. </summary>
		*
		* <returns>	The module manager. </returns>
		**************************************************************************************************/
	virtual const axis::services::management::GlobalProviderCatalog& GetModuleManager(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns plugin link information. </summary>
		*
		* <param name="index">	Zero-based index of the plugin link to obtain information. </param>
		*
		* <returns>	The plugin link information. </returns>
		**************************************************************************************************/
	virtual axis::services::management::PluginLink GetPluginLinkInformation(size_type index) const = 0;

	/**************************************************************************************************
		* <summary>	Returns how many plugin link has been established successfully. </summary>
		*
		* <returns>	The plugin link count. </returns>
		**************************************************************************************************/
	virtual size_type GetPluginLinkCount(void) const = 0;
protected:
  AxisApplication(void);
private:
  void initModuleManager(void);	// prepares the module manager to be used
};

} } } // namespace axis::application::runnable
