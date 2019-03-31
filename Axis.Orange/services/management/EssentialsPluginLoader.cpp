#include "EssentialsPluginLoader.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/factories/boundary_conditions/LockConstraintFactory.hpp"
#include "application/factories/boundary_conditions/NodalVelocityConstraintFactory.hpp"
#include "application/factories/boundary_conditions/PrescribedDisplacementConstraintFactory.hpp"
#include "application/factories/parsers/MultiLineCurveParserProvider.hpp"
#include "application/factories/parsers/NodalLoadParserProvider.hpp"
#include "application/locators/ConstraintParserLocator.hpp"
#include "system_messages.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aafbc = axis::application::factories::boundary_conditions;
namespace aal = axis::application::locators;
namespace asmg = axis::services::management;

asmg::EssentialsPluginLoader::EssentialsPluginLoader( void )
{
	// nothing to do here
}

asmg::EssentialsPluginLoader::~EssentialsPluginLoader( void )
{
	// force destroy factories if not done so before
	DestroyFactories();
}

void asmg::EssentialsPluginLoader::InitFactories( void )
{
	constraintFactories_.insert(new aafbc::LockConstraintFactory());
  constraintFactories_.insert(new aafbc::NodalVelocityConstraintFactory());
  constraintFactories_.insert(new aafbc::PrescribedDisplacementConstraintFactory());
  curveProviders_.insert(new aafp::MultiLineCurveParserProvider());
  loadProviders_.insert(new aafp::NodalLoadParserProvider());
}

void asmg::EssentialsPluginLoader::DestroyFactories( void ) 
{	
	constraint_factory_set::iterator csend = constraintFactories_.end();
	for (constraint_factory_set::iterator csit = constraintFactories_.begin(); csit != csend; ++csit)
	{
		delete *csit;
	}
	provider_set::iterator cvend = curveProviders_.end();
  for (provider_set::iterator cvit = curveProviders_.begin(); cvit != cvend; ++cvit)
  {
    delete *cvit;
  }
  provider_set::iterator ldend = loadProviders_.end();
  for (provider_set::iterator ldit = loadProviders_.begin(); ldit != ldend; ++ldit)
  {
    delete *ldit;
  }
  constraintFactories_.clear();
  curveProviders_.clear();
  loadProviders_.clear();
}

void asmg::EssentialsPluginLoader::StartPlugin( GlobalProviderCatalog& manager )
{
	// Register each provider in its respective locator or parent provider.
	// This will enable the program to parse and correctly build our objects.
	
	// first, we have to query the locator/providers location
  const char * constraintLocatorPath = ServiceLocator::GetNodalConstraintParserLocatorPath();
  const char * loadLocatorPath = ServiceLocator::GetLoadSectionInputParserProviderPath();
  const char * curveLocationPath = ServiceLocator::GetCurveSectionInputParserProviderPath();

	// now get the locator/provider itself; it is safe to static cast
	// because the program design guarantees it
	aal::ConstraintParserLocator& constraintLocator = 
    static_cast<aal::ConstraintParserLocator&>(manager.GetProvider(constraintLocatorPath));
  aafp::BlockProvider& parentLoadProvider = 
    static_cast<aafp::BlockProvider&>(manager.GetProvider(loadLocatorPath));
  aafp::BlockProvider& parentCurveProvider = 
    static_cast<aafp::BlockProvider&>(manager.GetProvider(curveLocationPath));
	
	// initialize and register our factories
	InitFactories();
	constraint_factory_set::iterator csend = constraintFactories_.end();
	for (constraint_factory_set::iterator csit = constraintFactories_.begin(); csit != csend; ++csit)
	{
		constraintLocator.RegisterFactory(**csit);
	}
  provider_set::iterator cvend = curveProviders_.end();
  for (provider_set::iterator cvit = curveProviders_.begin(); cvit != cvend; ++cvit)
  {
    parentCurveProvider.RegisterProvider(**cvit);
  }
  provider_set::iterator ldend = loadProviders_.end();
  for (provider_set::iterator ldit = loadProviders_.begin(); ldit != ldend; ++ldit)
  {
    parentLoadProvider.RegisterProvider(**ldit);
  }

  System::RegisterHook(AXIS_SYS_GPU_MEMORY_ARENA_INIT, hook_);
}

void asmg::EssentialsPluginLoader::UnloadPlugin( GlobalProviderCatalog& manager )
{
  System::UnregisterHook(hook_);

	// Unregister our parsers; that will disable the solver to read
	// and interpret our statements in the input file

  // first, we have to query the locator/providers location
  const char * constraintLocatorPath = ServiceLocator::GetNodalConstraintParserLocatorPath();
  const char * loadLocatorPath = ServiceLocator::GetLoadSectionInputParserProviderPath();
  const char * curveLocationPath = ServiceLocator::GetCurveSectionInputParserProviderPath();

  // now get the locator/provider itself; it is safe to static cast
  // because the program design guarantees it
  aal::ConstraintParserLocator& constraintLocator = 
    static_cast<aal::ConstraintParserLocator&>(manager.GetProvider(constraintLocatorPath));
  aafp::BlockProvider& parentLoadProvider = 
    static_cast<aafp::BlockProvider&>(manager.GetProvider(loadLocatorPath));
  aafp::BlockProvider& parentCurveProvider = 
    static_cast<aafp::BlockProvider&>(manager.GetProvider(curveLocationPath));

	// unregister factories
  constraint_factory_set::iterator csend = constraintFactories_.end();
  for (constraint_factory_set::iterator csit = constraintFactories_.begin(); csit != csend; ++csit)
  {
    constraintLocator.UnregisterFactory(**csit);
  }
  provider_set::iterator cvend = curveProviders_.end();
  for (provider_set::iterator cvit = curveProviders_.begin(); cvit != cvend; ++cvit)
  {
    parentCurveProvider.UnregisterProvider(**cvit);
  }
  provider_set::iterator ldend = loadProviders_.end();
  for (provider_set::iterator ldit = loadProviders_.begin(); ldit != ldend; ++ldit)
  {
    parentLoadProvider.UnregisterProvider(**ldit);
  }
	// destroy factories
	DestroyFactories();
}

void asmg::EssentialsPluginLoader::Destroy( void ) const
{
	delete this;
}

asmg::PluginInfo asmg::EssentialsPluginLoader::GetPluginInformation( void ) const
{
	return PluginInfo(_T("Axis Essentials"),
					          _T("Provides some basic functionalities commonly used in everyday analyses."),
					          _T("axis.my.libraries.essentials.2013.r1"),
					          _T("Renato T. Yamassaki"),
					          _T("(c) 2013 Renato T. Yamassaki"),
					          1, 1, 0, 680, 
					          _T(""));
}
