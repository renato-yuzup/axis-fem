#pragma once
#include <set>
#include "services/management/PluginLoader.hpp"
#include "application/factories/algorithms/SolverFactory.hpp"
#include "application/factories/algorithms/ClockworkFactory.hpp"
#include "MySolverHook.hpp"

namespace axis { namespace services { namespace management {

class MySolverPluginLoader : public axis::services::management::PluginLoader
{
public:
	MySolverPluginLoader(void);
	virtual ~MySolverPluginLoader(void);
	virtual void StartPlugin( GlobalProviderCatalog& manager );
	virtual void UnloadPlugin( GlobalProviderCatalog& manager );
	virtual void Destroy( void ) const;
	virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
private:
  typedef std::set<axis::application::factories::algorithms::ClockworkFactory *> clockwork_factory_set;
  typedef std::set<axis::application::factories::algorithms::SolverFactory *> solver_factory_set;

  void InitFactories(void);
  void DestroyFactories(void);

  clockwork_factory_set clockworks_;
  solver_factory_set solvers_;
  MySolverHook hook_;
};

} } } // namespace axis::services::management
