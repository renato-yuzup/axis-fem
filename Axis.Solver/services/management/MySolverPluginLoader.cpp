#include "MySolverPluginLoader.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/factories/algorithms/LinearStaticCGSolverFactory.hpp"
#include "application/factories/algorithms/ExplicitStandardTimeSolverFactory.hpp"
#include "application/factories/algorithms/RegularClockworkFactory.hpp"
#include "application/factories/algorithms/WaveSpeedProportionalClockworkFactory.hpp"
#include "application/locators/SolverFactoryLocator.hpp"
#include "application/locators/ClockworkFactoryLocator.hpp"
#include "system_messages.hpp"

namespace aafa = axis::application::factories::algorithms;
namespace aal = axis::application::locators;
namespace asmg = axis::services::management;

asmg::MySolverPluginLoader::MySolverPluginLoader( void )
{
	// nothing to do here
}

asmg::MySolverPluginLoader::~MySolverPluginLoader( void )
{
	// nothing to do here
}

void asmg::MySolverPluginLoader::InitFactories( void )
{
  solvers_.insert(new aafa::LinearStaticCGSolverFactory());			  // linear static solver
  solvers_.insert(new aafa::ExplicitStandardTimeSolverFactory()); // standard explicit solver

  // clockwork factories
  clockworks_.insert(new aafa::RegularClockworkFactory());
  clockworks_.insert(new aafa::WaveSpeedProportionalClockworkFactory());
}

void asmg::MySolverPluginLoader::DestroyFactories( void )
{
  solver_factory_set::iterator send = solvers_.end();
  for (solver_factory_set::iterator sit = solvers_.begin(); sit != send; ++sit)
  {
    delete *sit;
  }
  clockwork_factory_set::iterator cend = clockworks_.end();
  for (clockwork_factory_set::iterator cit = clockworks_.begin(); cit != cend; ++cit)
  {
    delete *cit;
  }
	solvers_.clear();
  clockworks_.clear();
}

void asmg::MySolverPluginLoader::StartPlugin( GlobalProviderCatalog& manager )
{
	// Create material factories
	InitFactories();

	// now, get a reference to the locator
	// (don't worry, the cast below is guaranteed by design)
  aal::SolverFactoryLocator& solverLocator       = static_cast<aal::SolverFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetSolverLocatorPath()));
  aal::ClockworkFactoryLocator& clockworkLocator = static_cast<aal::ClockworkFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetClockworkFactoryLocatorPath()));

  // solver factories
  solver_factory_set::iterator send = solvers_.end();
  for (solver_factory_set::iterator sit = solvers_.begin(); sit != send; ++sit)
  {
    solverLocator.RegisterFactory(**sit);
  }
  clockwork_factory_set::iterator cend = clockworks_.end();
  for (clockwork_factory_set::iterator cit = clockworks_.begin(); cit != cend; ++cit)
  {
    clockworkLocator.RegisterFactory(**cit);
  }

  System::RegisterHook(AXIS_SYS_GPU_MEMORY_ARENA_INIT, hook_);
}

void asmg::MySolverPluginLoader::UnloadPlugin( GlobalProviderCatalog& manager )
{
  System::UnregisterHook(hook_);

  // get a reference to the locator
  aal::SolverFactoryLocator& solverLocator       = static_cast<aal::SolverFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetSolverLocatorPath()));
  aal::ClockworkFactoryLocator& clockworkLocator = static_cast<aal::ClockworkFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetClockworkFactoryLocatorPath()));

  // solver factories
  solver_factory_set::iterator send = solvers_.end();
  for (solver_factory_set::iterator sit = solvers_.begin(); sit != send; ++sit)
  {
    solverLocator.UnregisterFactory(**sit);
  }
  clockwork_factory_set::iterator cend = clockworks_.end();
  for (clockwork_factory_set::iterator cit = clockworks_.begin(); cit != cend; ++cit)
  {
    clockworkLocator.UnregisterFactory(**cit);
  }

	// destroy factories
	DestroyFactories();
}

void asmg::MySolverPluginLoader::Destroy( void ) const
{
	delete this;
}

asmg::PluginInfo asmg::MySolverPluginLoader::GetPluginInformation( void ) const
{
	return PluginInfo(_T("axis Standard materials Library"),
					  _T("Provides a basic set of material models for use with axis solver."),
					  _T("axis.my.libraries.materials.2012.r1"),
					  _T("Renato T. Yamassaki"),
					  _T("(c) 2012 Renato T. Yamassaki"),
					  1, 0, 0, 100, 
					  _T(""));
}
