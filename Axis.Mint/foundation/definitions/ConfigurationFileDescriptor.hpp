#pragma once
#include "AxisString.hpp"
#include "foundation/Axis.Mint.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API ConfigurationFileDescriptor
{
private:
	ConfigurationFileDescriptor(void);
public:
	~ConfigurationFileDescriptor(void);

	static const axis::String::char_type * PluginsSectionName;
	static const axis::String::char_type * CustomSystemPluginsSectionName;
	static const axis::String::char_type * NonVolatilePluginsSectionName;
	static const axis::String::char_type * VolatilePluginsSectionName;

	static const axis::String::char_type * PluginDescriptorTypeName;
	static const axis::String::char_type * PluginDirectoryDescriptorName;
	static const axis::String::char_type * PluginLocationAttributeName;
	static const axis::String::char_type * PluginDirectoryLocationAttributeName;
	static const axis::String::char_type * PluginDirectoryRecursiveSearchAttributeName;
	static const axis::String::char_type * RootSectionName;
};		

} } } // namespace axis::foundation::definitions
