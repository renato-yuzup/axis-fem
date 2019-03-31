#pragma once

#include "ConfigurationScript.hpp"
#include "../io/StreamReader.hpp"
#include <map>
#define TIXML_USE_STL
#include "tinyxml/tinyxml.h"

namespace axis
{
	namespace services
	{
		namespace configuration
		{
			class XmlConfigurationScript: public ConfigurationScript
			{
			private:
				typedef std::map<TiXmlNode *, ConfigurationScript *> section_tree;
				typedef TiXmlDocument settings_tree;
				
				axis::String _sectionName;
				settings_tree& _settings;
				section_tree _sections;
				axis::String _rootNodeName;
				TiXmlNode *_root;
				 

				XmlConfigurationScript(TiXmlNode *root, const axis::String& sectionName);
				XmlConfigurationScript(void);

				void wipe_sections(void);

				ConfigurationScript& create_dependent_instance(TiXmlNode *root, const axis::String& sectionName);

				int get_node_order_number(TiXmlNode *node) const;
			public:
				static XmlConfigurationScript& ReadFromFile(const axis::String& fileName);
				
				XmlConfigurationScript(const XmlConfigurationScript& script);
				~XmlConfigurationScript(void);

				virtual ConfigurationScript& GetSection( const axis::String& sectionName );
				virtual bool ContainsSection(const axis::String& sectionName);

				virtual ConfigurationScript& GetFirstChildSection(void);
				virtual bool HasChildSections(void);

				virtual ConfigurationScript& GetNextSiblingSection(void);
				virtual bool HasMoreSiblingsSection(void);

				virtual ConfigurationScript *FindNextSibling(const axis::String& sectionName);


				virtual bool ContainsAttribute(const axis::String& attributeName);
				virtual axis::String GetAttributeValue( const axis::String attributeName );
				virtual axis::String GetAttributeWithDefault(const axis::String& attributeName, const axis::String& defaultValue);
				
				virtual axis::String GetConfigurationText(void);
				virtual bool ContainsConfigurationText(void);

				virtual axis::String GetSectionName(void) const;

				virtual axis::String GetConfigurationPath(void) const;

				virtual void Destroy( void ) const;
			};
		}
	}

}