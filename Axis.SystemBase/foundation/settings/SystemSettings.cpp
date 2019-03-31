#include "SystemSettings.hpp"

using namespace axis::foundation::settings;


axis::foundation::settings::SystemSettings::SystemSettings( void )
{
	// just to disallow instantiation of this class
}


// Initialize all static members

#ifdef _UNICODE
	const wchar_t * SystemSettings::DefaultConfigurationFile = L"axis.config";
	const wchar_t * SystemSettings::LocaleFolderName = L"Locales";
	#ifdef _WINDOWS
		const wchar_t * SystemSettings::NewLine = L"\n";
	#else
		const wchar_t * SystemSettings::NewLine = L"\r";
	#endif
#else
	const char * SystemSettings::DefaultConfigurationFile = "axis.config";
	const char * SystemSettings::LocaleFolderName = "Locales";
	#ifdef _WINDOWS
		const char * SystemSettings::NewLine = "\n";
	#else
		const char * SystemSettings::NewLine = "\r";
	#endif
#endif
