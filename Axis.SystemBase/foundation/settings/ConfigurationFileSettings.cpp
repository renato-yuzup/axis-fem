#include "ConfigurationFileSettings.hpp"

using namespace axis::foundation::settings;

axis::foundation::settings::ConfigurationFileSettings::ConfigurationFileSettings( void )
{
	// just to disallow instantiation of this class
}

#ifdef _UNICODE
	const wchar_t * ConfigurationFileSettings::ConfigurationRoot = L"axisConfiguration";
	const wchar_t * ConfigurationFileSettings::DirectoriesSection = L"directories";
	const wchar_t * ConfigurationFileSettings::DirectoriesElements = L"path";
	const wchar_t * ConfigurationFileSettings::DirectoriesElementsFeatureAttr = L"feature";
	const wchar_t * ConfigurationFileSettings::DirectoriesElementsLocationAttr = L"location";
#else
	const char * ConfigurationFileSettings::ConfigurationRoot = "axisConfiguration";
	const char * ConfigurationFileSettings::DirectoriesSection = "directories";
	const char * ConfigurationFileSettings::DirectoriesElements = "path";
	const char * ConfigurationFileSettings::DirectoriesElementsFeatureAttr = "feature";
	const char * ConfigurationFileSettings::DirectoriesElementsLocationAttr = "location";
#endif

