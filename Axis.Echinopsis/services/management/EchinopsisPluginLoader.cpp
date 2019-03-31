#include "StdAfx.h"
#include "EchinopsisPluginLoader.hpp"
#include "EchinopsisPluginLoader_Pimpl.hpp"
#include "application/factories/collectors/MatlabDatasetCollectorFactory.hpp"
#include "application/factories/workbooks/MatlabDatasetWorkbookFactory.hpp"
#include "application/factories/collectors/GeneralSummaryNodeCollectorFactory.hpp"
#include "application/factories/collectors/GeneralSummaryElementCollectorFactory.hpp"
#include "application/locators/CollectorFactoryLocator.hpp"
#include "application/locators/WorkbookFactoryLocator.hpp"
#include "services/management/ServiceLocator.hpp"

namespace asmg = axis::services::management;
namespace aafc = axis::application::factories::collectors;
namespace aafw = axis::application::factories::workbooks;
namespace aal = axis::application::locators;
namespace asmg = axis::services::management;

axis::services::management::EchinopsisPluginLoader::EchinopsisPluginLoader( void )
{
  pimpl_ = new Pimpl();
  loadOk_ = false;
}

asmg::EchinopsisPluginLoader::~EchinopsisPluginLoader( void )
{
  delete pimpl_;
}

void asmg::EchinopsisPluginLoader::StartPlugin( GlobalProviderCatalog& manager )
{
  // get feature locators
  const char *collectorFactoryPath = asmg::ServiceLocator::GetCollectorFactoryLocatorPath();
  const char *workbookFactoryPath = asmg::ServiceLocator::GetWorkbookFactoryLocatorPath();

  bool ok = EnsurePreRequisites(manager);
  if (ok)
  {
    CreateFactories(manager);

    // load our factories
    aal::CollectorFactoryLocator& collectorLocator = static_cast<aal::CollectorFactoryLocator&>(
                                                          manager.GetProvider(collectorFactoryPath));
    aal::WorkbookFactoryLocator& workbookLocator = static_cast<aal::WorkbookFactoryLocator&>(
                                                          manager.GetProvider(workbookFactoryPath));
    collectorLocator.RegisterFactory(*pimpl_->MatlabCollectorFactory);
    workbookLocator.RegisterFactory(*pimpl_->MatlabFactory);
    collectorLocator.RegisterFactory(*pimpl_->ReportElementCollectorFactory);
    collectorLocator.RegisterFactory(*pimpl_->ReportNodeCollectorFactory);
    workbookLocator.RegisterFactory(*pimpl_->ReportFactory);
    collectorLocator.RegisterFactory(*pimpl_->HyperworksElementCollectorFactory);
    collectorLocator.RegisterFactory(*pimpl_->HyperworksNodeCollectorFactory);
    workbookLocator.RegisterFactory(*pimpl_->HyperworksFactory);
  }
  loadOk_ = ok;
}

void asmg::EchinopsisPluginLoader::Destroy( void ) const
{
  delete this;
}

void asmg::EchinopsisPluginLoader::UnloadPlugin( GlobalProviderCatalog& manager )
{
  if (!loadOk_) return;

  const char *collectorFactoryPath = asmg::ServiceLocator::GetCollectorFactoryLocatorPath();
  const char *workbookFactoryPath = asmg::ServiceLocator::GetWorkbookFactoryLocatorPath();
  aal::CollectorFactoryLocator& collectorLocator = static_cast<aal::CollectorFactoryLocator&>(
    manager.GetProvider(collectorFactoryPath));
  aal::WorkbookFactoryLocator& workbookLocator = static_cast<aal::WorkbookFactoryLocator&>(
    manager.GetProvider(workbookFactoryPath));

  collectorLocator.UnregisterFactory(*pimpl_->MatlabCollectorFactory);
  workbookLocator.UnregisterFactory(*pimpl_->MatlabFactory);
  collectorLocator.UnregisterFactory(*pimpl_->ReportElementCollectorFactory);
  collectorLocator.UnregisterFactory(*pimpl_->ReportNodeCollectorFactory);
  workbookLocator.UnregisterFactory(*pimpl_->ReportFactory);
  collectorLocator.UnregisterFactory(*pimpl_->HyperworksElementCollectorFactory);
  collectorLocator.UnregisterFactory(*pimpl_->HyperworksNodeCollectorFactory);
  workbookLocator.UnregisterFactory(*pimpl_->HyperworksFactory);
}

asmg::PluginInfo asmg::EchinopsisPluginLoader::GetPluginInformation( void ) const
{
  return PluginInfo(_T("Axis Echinopsis Library"),
    _T("Library containing standard output collectors."),
    _T("axis.Echinopsis"),
    _T("Renato T. Yamassaki"),
    _T("(c) Copyright 2013"),
    0, 8, 0, 5, 
    _T(""));
}

bool asmg::EchinopsisPluginLoader::EnsurePreRequisites( GlobalProviderCatalog& manager ) const
{
  // get feature locators
  const char *collectorFactoryPath = asmg::ServiceLocator::GetCollectorFactoryLocatorPath();
  const char *workbookFactoryPath = asmg::ServiceLocator::GetWorkbookFactoryLocatorPath();

  return manager.ExistsProvider(collectorFactoryPath) && 
         manager.ExistsProvider(workbookFactoryPath);
}

void asmg::EchinopsisPluginLoader::CreateFactories( GlobalProviderCatalog& manager ) const
{
  pimpl_->MatlabCollectorFactory = new aafc::MatlabDatasetCollectorFactory();
  pimpl_->MatlabFactory = new aafw::MatlabDatasetWorkbookFactory();

  pimpl_->ReportElementCollectorFactory = new aafc::TextReportElementCollectorFactory();
  pimpl_->ReportNodeCollectorFactory = new aafc::TextReportNodeCollectorFactory();
  pimpl_->ReportFactory = new aafw::TextReportWorkbookFactory();

  pimpl_->HyperworksElementCollectorFactory = new aafc::HyperworksElementCollectorFactory();
  pimpl_->HyperworksNodeCollectorFactory = new aafc::HyperworksNodeCollectorFactory();
  pimpl_->HyperworksFactory = new aafw::HyperworksWorkbookFactory();
}
