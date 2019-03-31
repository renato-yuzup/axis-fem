#include "MyElementsPluginLoader.hpp"
#include "application/factories/parsers/NonLinearHexahedronSimpleParserFactory.hpp"
#include "application/factories/parsers/NonLinearHexahedronFlaBelytschkoParserFactory.hpp"
#include "application/factories/parsers/LinearHexahedronSimpleParserFactory.hpp"
#include "application/factories/parsers/LinearHexahedronFlaBelytschkoParserFactory.hpp"
#include "application/factories/parsers/LinearHexahedronPusoParserFactory.hpp"
#include "services/management/ServiceLocator.hpp"
#include "system_messages.hpp"
#include "System.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aal = axis::application::locators;
namespace asmg = axis::services::management;

asmg::MyElementsPluginLoader::MyElementsPluginLoader( void )
{
	// nothing to do here
}

asmg::MyElementsPluginLoader::~MyElementsPluginLoader( void )
{
	// force destroy factories if not done so before
	DestroyFactories();
}

void asmg::MyElementsPluginLoader::InitFactories( aal::ElementParserLocator& locator )
{
	_ourFactories.insert(new aafp::LinearHexahedronSimpleParserFactory(locator));
  _ourFactories.insert(new aafp::LinearHexahedronFlaBelytschkoParserFactory(locator));
  _ourFactories.insert(new aafp::NonLinearHexahedronFlaBelytschkoParserFactory(locator));
  _ourFactories.insert(new aafp::LinearHexahedronPusoParserFactory(locator));
  _ourFactories.insert(new aafp::NonLinearHexahedronSimpleParserFactory(locator));
}

void asmg::MyElementsPluginLoader::DestroyFactories( void ) 
{
	// destroy element factories
	factory_set::iterator end = _ourFactories.end();
	for (factory_set::iterator it = _ourFactories.begin(); it != end; ++it)
	{
		(*it)->Destroy();
	}
	_ourFactories.clear();
}

void asmg::MyElementsPluginLoader::StartPlugin( GlobalProviderCatalog& manager )
{
	// Register in element locator our parsers; that will enable
	// the solver to read and interpret element statements in the
	// input file, loading them into memory as objects
	
	// first, we have to query where the element locator is in the
	// plugins namespace
	const char * locatorPath = ServiceLocator::GetFiniteElementLocatorPath();

	// now get the locator itself; it is safe to static cast
	// because the program design guarantees it
	aal::ElementParserLocator& locator = 
    static_cast<aal::ElementParserLocator&>(manager.GetProvider(locatorPath));
	
	// initialize our element factories
	InitFactories(locator);

	// register factories
	factory_set::iterator end = _ourFactories.end();
	for (factory_set::iterator it = _ourFactories.begin(); it != end; ++it)
	{
		locator.RegisterFactory(**it);
	}

  System::RegisterHook(AXIS_SYS_GPU_MEMORY_ARENA_INIT, hook_);
}

void asmg::MyElementsPluginLoader::UnloadPlugin( GlobalProviderCatalog& manager )
{
  System::UnregisterHook(hook_);

	// Unregister our parsers; that will disable the solver to read
	// and interpret element statements in the input file

	// first, we have to query where the element locator is in the
	// plugins namespace
	const char * locatorPath = ServiceLocator::GetFiniteElementLocatorPath();

	// now get the locator itself; it is safe to static cast
	// because the program design guarantees it
	aal::ElementParserLocator& locator = 
    static_cast<aal::ElementParserLocator&>(manager.GetProvider(locatorPath));

	// unregister factories
	factory_set::iterator end = _ourFactories.end();
	for (factory_set::iterator it = _ourFactories.begin(); it != end; ++it)
	{
		locator.UnregisterFactory(**it);
	}

	// destroy factories
	DestroyFactories();
}

void asmg::MyElementsPluginLoader::Destroy( void ) const
{
	delete this;
}

asmg::PluginInfo asmg::MyElementsPluginLoader::GetPluginInformation( void ) const
{
	return PluginInfo(_T("Axis Standard Elements Library"),
					          _T("Provides a basic set of finite elements for use with Axis Solver."),
					          _T("axis.my.libraries.elements.2013.r1"),
					          _T("Renato T. Yamassaki"),
					          _T("(c) 2013 Renato T. Yamassaki"),
					          1, 1, 0, 680, 
					          _T(""));
}
