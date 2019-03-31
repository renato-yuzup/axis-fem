#include <windows.h>
#include "WindowsPluginConnector.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/ApplicationErrorException.hpp"

namespace asmg = axis::services::management;

const char * PluginEntryPointFunctionName = "AxisPluginLoader_GetLoader";

extern "C"
{
	typedef asmg::PluginLoader * (*PluginLoaderFunction)(void);
};

// This is our auxiliary function
void UnloadWindowsLibrary( HMODULE hMod )
{
	if (!FreeLibrary(hMod))
	{	// failure in unloading DLL -- this is REALLY bad -- throw a severe exception
		throw axis::foundation::ApplicationErrorException(
      _T("Application condition has been jeopardized due to a dangling module on a failed unload procedure."));
	}
}

asmg::WindowsPluginConnector::WindowsPluginConnector( const axis::String& dllFileName )
{
	_dllPath = dllFileName;
	_lastError = kNoError;
	_moduleHandle = NULL;
	_loader = NULL;
	_pluginLoaded = false;
	_isRegistered = false;
}

asmg::WindowsPluginConnector::~WindowsPluginConnector( void )
{
	// nothing to do here
}

void asmg::WindowsPluginConnector::LoadPlugin( void )
{
	// ignore if plugin is already loaded
	if (_pluginLoaded) return;

	// first, try to load the library into this application address
	// space
	HMODULE hMod = LoadLibrary(_dllPath.c_str());
	if (hMod == NULL)
	{	// failure loading the DLL
		_lastError = kInitializeError;
		return;
	}

	// now, try to find the entry point function which will
	// instantiate the loader
	FARPROC funcAddr = GetProcAddress(hMod, PluginEntryPointFunctionName);
	if (funcAddr == NULL)
	{	// entry point not found
		_lastError = kEntryPointNotFound;

		// unload DLL
		UnloadWindowsLibrary(hMod);
		return;
	}

	// good, we now have the entry point -- try to get the loader
	asmg::PluginLoader *loader = NULL;
	try
	{
		PluginLoaderFunction loaderFun = (PluginLoaderFunction)funcAddr;
		loader = loaderFun();
	}
	catch (axis::foundation::AxisException&)
	{
		// the plugin loader function threw a known exception type,
		// which might state that it really is a plugin module, but
		// failed to load anyway
		_lastError = kLoaderException;

		// unload library
		UnloadWindowsLibrary(hMod);
		return;
	}
	catch (...)
	{	// another exception type was thrown -- strange behavior
		_lastError = kUnexpectedLoaderBehavior;

		// unload library
		UnloadWindowsLibrary(hMod);
		return;
	}

	// loader object retrieved successfully, update object state
	_moduleHandle = hMod;
	_lastError = kNoError;
	_loader = loader;
	_pluginLoaded = true;
}

bool asmg::WindowsPluginConnector::IsPluginLoaded( void ) const
{
	return _pluginLoaded;
}

void asmg::WindowsPluginConnector::RegisterPlugin( GlobalProviderCatalog& manager )
{
	if (!IsPluginLoaded())
	{
		throw axis::foundation::InvalidOperationException();
	}
	if (IsPluginReady())
	{	// cannot register again
		throw axis::foundation::InvalidOperationException();
	}
	// run plugin loader (any exceptions should be handled by the
	// caller)
	_loader->StartPlugin(manager);
	_isRegistered = true;
}

asmg::PluginConnector::ErrorReason asmg::WindowsPluginConnector::GetErrorCondition( void ) const
{
	return _lastError;
}

void asmg::WindowsPluginConnector::Destroy( void ) const
{
	delete this;
}

axis::String asmg::WindowsPluginConnector::GetFileName( void ) const
{
	return _dllPath;
}

bool asmg::WindowsPluginConnector::IsPluginReady( void ) const
{
	return _isRegistered;
}

void asmg::WindowsPluginConnector::UnloadPlugin( GlobalProviderCatalog& manager )
{
	if (!IsPluginLoaded())
	{
		throw axis::foundation::InvalidOperationException();
	}
	if (IsPluginReady())
	{
		_loader->UnloadPlugin(manager);
	}
	// destroy loader object
	_loader->Destroy();

	// unload library
	UnloadWindowsLibrary(static_cast<HMODULE>(_moduleHandle));

	// update state
	_pluginLoaded = false;
	_lastError = kNoError;
	_moduleHandle = 0;
	_loader = NULL;
}

asmg::PluginInfo asmg::WindowsPluginConnector::GetPluginInformation( void ) const
{
	if (!IsPluginReady() || !IsPluginLoaded())
	{
		throw axis::foundation::InvalidOperationException(_T("Plugin not ready."));
	}
	return _loader->GetPluginInformation();
}
