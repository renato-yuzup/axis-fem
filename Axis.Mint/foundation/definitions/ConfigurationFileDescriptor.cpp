#include "ConfigurationFileDescriptor.hpp"

namespace afd = axis::foundation::definitions;

afd::ConfigurationFileDescriptor::ConfigurationFileDescriptor( void )
{
	// nothing to do here
}

afd::ConfigurationFileDescriptor::~ConfigurationFileDescriptor( void )
{
	// nothing to do here
}

const axis::String::char_type * afd::ConfigurationFileDescriptor::RootSectionName = _T("axis.settings");

const axis::String::char_type * afd::ConfigurationFileDescriptor::PluginDirectoryRecursiveSearchAttributeName = _T("recursive");
const axis::String::char_type * afd::ConfigurationFileDescriptor::PluginDirectoryLocationAttributeName = _T("location");
const axis::String::char_type * afd::ConfigurationFileDescriptor::PluginLocationAttributeName = _T("location");
const axis::String::char_type * afd::ConfigurationFileDescriptor::PluginDirectoryDescriptorName = _T("library");
const axis::String::char_type * afd::ConfigurationFileDescriptor::PluginDescriptorTypeName = _T("plugin");

const axis::String::char_type * afd::ConfigurationFileDescriptor::VolatilePluginsSectionName = _T("volatile_plugins");
const axis::String::char_type * afd::ConfigurationFileDescriptor::NonVolatilePluginsSectionName = _T("userbase_plugins");
const axis::String::char_type * afd::ConfigurationFileDescriptor::CustomSystemPluginsSectionName = _T("system_plugins");
const axis::String::char_type * afd::ConfigurationFileDescriptor::PluginsSectionName = _T("plugins");

