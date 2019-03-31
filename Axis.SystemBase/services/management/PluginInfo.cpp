#include "PluginInfo.hpp"


axis::services::management::PluginInfo::PluginInfo( const axis::String& pluginName, 
													const axis::String& pluginDescription, 
													const axis::String& internalName, 
													const axis::String& authorName, 
													const axis::String& copyrightNotice, 
													int pluginMajorVersionNumber, 
													int pluginMinorVersionNumber, 
													int pluginRevisionVersionNumber, 
													int pluginBuildVersionNumber, 
													const axis::String& versionReleaseString )
{
	_name = pluginName;
	_description = pluginDescription;
	_internalName = internalName;
	_author = authorName;
	_copyright = copyrightNotice;
	_versionString = versionReleaseString;
	_major = pluginMajorVersionNumber;
	_minor = pluginMinorVersionNumber;
	_revision = pluginRevisionVersionNumber;
	_build = pluginBuildVersionNumber;
}

axis::String axis::services::management::PluginInfo::GetPluginName( void ) const
{
	return _name;
}

axis::String axis::services::management::PluginInfo::GetPluginDescription( void ) const
{
	return _description;
}

axis::String axis::services::management::PluginInfo::GetPluginInternalName( void ) const
{
	return _internalName;
}

axis::String axis::services::management::PluginInfo::GetPluginAuthor( void ) const
{
	return _author;
}

axis::String axis::services::management::PluginInfo::GetPluginCopyrightNotice( void ) const
{
	return _copyright;
}

axis::String axis::services::management::PluginInfo::GetPluginVersionReleaseString( void ) const
{
	return _versionString;
}

int axis::services::management::PluginInfo::GetPluginMajorVersion( void ) const
{
	return _major;
}

int axis::services::management::PluginInfo::GetPluginMinorVersion( void ) const
{
	return _minor;
}

int axis::services::management::PluginInfo::GetPluginRevisionVersion( void ) const
{
	return _revision;
}

int axis::services::management::PluginInfo::GetPluginBuildVersion( void ) const
{
	return _build;
}
