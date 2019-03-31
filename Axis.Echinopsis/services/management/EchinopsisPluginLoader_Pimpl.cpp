#include "EchinopsisPluginLoader_Pimpl.hpp"

axis::services::management::EchinopsisPluginLoader::Pimpl::Pimpl( void )
{
  MatlabCollectorFactory = NULL;
  MatlabFactory  = NULL;
  ReportNodeCollectorFactory = NULL;
  ReportElementCollectorFactory = NULL;
  ReportFactory = NULL;
}

axis::services::management::EchinopsisPluginLoader::Pimpl::~Pimpl( void )
{
  if (MatlabCollectorFactory != NULL) MatlabCollectorFactory->Destroy();
  if (MatlabFactory != NULL) MatlabFactory->Destroy();
  if (ReportFactory != NULL) ReportFactory->Destroy();
  if (ReportElementCollectorFactory != NULL) ReportElementCollectorFactory->Destroy();
  if (ReportNodeCollectorFactory != NULL) ReportNodeCollectorFactory->Destroy();
  if (HyperworksFactory != NULL) HyperworksFactory->Destroy();
  if (HyperworksElementCollectorFactory != NULL) HyperworksElementCollectorFactory->Destroy();
  if (HyperworksNodeCollectorFactory != NULL) HyperworksNodeCollectorFactory->Destroy();
  MatlabCollectorFactory = NULL;
  MatlabFactory  = NULL;
  ReportNodeCollectorFactory = NULL;
  ReportElementCollectorFactory = NULL;
  ReportFactory = NULL;
  HyperworksNodeCollectorFactory = NULL;
  HyperworksElementCollectorFactory = NULL;
  HyperworksFactory = NULL;
}
