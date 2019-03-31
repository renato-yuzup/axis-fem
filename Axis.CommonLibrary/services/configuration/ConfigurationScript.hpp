#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace configuration
		{
			class AXISCOMMONLIBRARY_API ConfigurationScript
			{
			public:
				static ConfigurationScript& ReadFromXml(const axis::String& filename);

				virtual ~ConfigurationScript(void);

				virtual ConfigurationScript& GetSection(const axis::String& sectionName) = 0;
				virtual bool ContainsSection(const axis::String& sectionName) = 0;

				virtual ConfigurationScript& GetFirstChildSection(void) = 0;
				virtual bool HasChildSections(void) = 0;

				virtual ConfigurationScript& GetNextSiblingSection(void) = 0;
				virtual bool HasMoreSiblingsSection(void) = 0;

				virtual ConfigurationScript *FindNextSibling(const axis::String& sectionName) = 0;

				virtual axis::String GetAttributeValue(const axis::String attributeName) = 0;
				virtual axis::String GetAttributeWithDefault(const axis::String& attributeName, const axis::String& defaultValue) = 0;
				virtual bool ContainsAttribute(const axis::String& attributeName) = 0;

				virtual axis::String GetConfigurationText(void) = 0;
				virtual bool ContainsConfigurationText(void) = 0;

				virtual axis::String GetSectionName(void) const = 0;

				virtual axis::String GetConfigurationPath(void) const = 0;

				virtual void Destroy(void) const = 0;
			};
		}
	}
}