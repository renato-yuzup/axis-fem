#pragma once
#include "services/management/PluginLoader.hpp"

namespace axis { namespace services { namespace management {

class EchinopsisPluginLoader : public axis::services::management::PluginLoader
{
public:
  EchinopsisPluginLoader(void);
  virtual ~EchinopsisPluginLoader(void);

  virtual void StartPlugin( GlobalProviderCatalog& manager );

  virtual void Destroy( void ) const;

  virtual void UnloadPlugin( GlobalProviderCatalog& manager );

  virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
private:
  class Pimpl;

  bool EnsurePreRequisites(GlobalProviderCatalog& manager) const;
  void CreateFactories(GlobalProviderCatalog& manager) const;
  Pimpl *pimpl_;
  bool loadOk_;
};

} } } // namespace axis::services::management

