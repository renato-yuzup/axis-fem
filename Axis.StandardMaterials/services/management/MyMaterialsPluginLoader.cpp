#include "MyMaterialsPluginLoader.hpp"

#include "application/locators/MaterialFactoryLocator.hpp"
#include "application/factories/materials/BiLinearPlasticityModelFactory.hpp"
#include "application/factories/materials/LinearIsoElasticFactory.hpp"
#include "application/factories/materials/NeoHookeanModelFactory.hpp"

#include "services/management/ServiceLocator.hpp"
#include "system_messages.hpp"
#include "System.hpp"

namespace aal = axis::application::locators;
namespace aafm = axis::application::factories::materials;
namespace asmg = axis::services::management;

asmg::MyMaterialsPluginLoader::MyMaterialsPluginLoader( void )
{
	// nothing to do here
}

asmg::MyMaterialsPluginLoader::~MyMaterialsPluginLoader( void )
{
	// nothing to do here
}

void asmg::MyMaterialsPluginLoader::LoadMaterialFactories( void )
{
  _materialFactories.insert(new aafm::LinearIsoElasticFactory());
  _materialFactories.insert(new aafm::NeoHookeanModelFactory());
  _materialFactories.insert(new aafm::BiLinearPlasticityModelFactory());
}

void asmg::MyMaterialsPluginLoader::UnloadMaterialFactories( void )
{
	factory_set::iterator end = _materialFactories.end();
	for (factory_set::iterator it = _materialFactories.begin(); it != end; ++it)
	{
		(*it)->Destroy();
	}
	_materialFactories.clear();
}

void asmg::MyMaterialsPluginLoader::StartPlugin(GlobalProviderCatalog& manager)
{
	// Create material factories
	LoadMaterialFactories();

	// Register our material factories into the standard material
	// factory locator of the program. To do so, we need first to
	// get the actual plugin path of the locator.
	const char * materialLocatorPath = ServiceLocator::GetMaterialFactoryLocatorPath();

	// now, get a reference to the locator
	// (don't worry, the cast below is guaranteed by design)
	aal::MaterialFactoryLocator& locator = 
    static_cast<aal::MaterialFactoryLocator&>(
    manager.GetProvider(materialLocatorPath));

	// register factories
	factory_set::iterator end = _materialFactories.end();
	for (factory_set::iterator it = _materialFactories.begin(); it != end; ++it)
	{
		locator.RegisterFactory(**it);
	}
  System::RegisterHook(AXIS_SYS_GPU_MEMORY_ARENA_INIT, hook_);
}

void asmg::MyMaterialsPluginLoader::UnloadPlugin( GlobalProviderCatalog& manager )
{
  System::UnregisterHook(hook_);

	// Unregister our material factories from the standard material
	// factory locator of the program. Doing so, will actually
	// disconnect our plugin from the program.
	const char * materialLocatorPath = ServiceLocator::GetMaterialFactoryLocatorPath();

	// now, get a reference to the locator
	// (don't worry, the cast below is guaranteed by design)
	aal::MaterialFactoryLocator& locator = 
    static_cast<aal::MaterialFactoryLocator&>(
    manager.GetProvider(materialLocatorPath));

	// unregister factories
	factory_set::iterator end = _materialFactories.end();
	for (factory_set::iterator it = _materialFactories.begin(); it != end; ++it)
	{
		locator.UnregisterFactory(**it);
	}

	// destroy factories
	UnloadMaterialFactories();
}

void asmg::MyMaterialsPluginLoader::Destroy( void ) const
{
	delete this;
}

asmg::PluginInfo asmg::MyMaterialsPluginLoader::GetPluginInformation( void ) const
{
	return PluginInfo(_T("axis Standard materials Library"),
					  _T("Provides a basic set of material models for use with axis solver."),
					  _T("axis.my.libraries.materials.2012.r1"),
					  _T("Renato T. Yamassaki"),
					  _T("(c) 2012 Renato T. Yamassaki"),
					  1, 0, 0, 100, 
					  _T(""));
}
