#include "XmlConfigurationScript.hpp"

#include <sstream>
#include <fstream>
#include <boost/lexical_cast.hpp>

// This is the source code of TinyXML, the core XML engine. 
// Some modifications were made in order to support Unicode filenames.
// Many thanks to Lee Thomason!
#include <tinyxml/tinyxml.cpp>
#include <tinyxml/tinyxmlerror.cpp>
#include <tinyxml/tinyxmlparser.cpp>

#include "string_traits.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/SyntaxMismatchException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/IOException.hpp"
#include "foundation/ConfigurationNotFoundException.hpp"


#define TIXML_ERROR_DOCUMENT_EMPTY		12	// error due to empty document

#ifdef _UNICODE	
	typedef wchar_t string_unit;
	typedef std::wifstream input_stream;
#else
	typedef char string_unit;
	typedef std::ifstream input_stream;
#endif

axis::services::configuration::XmlConfigurationScript::XmlConfigurationScript( void ) : _settings(*new settings_tree())
{
	// do nothing
}

axis::services::configuration::XmlConfigurationScript::XmlConfigurationScript( const XmlConfigurationScript& script ) : _settings(*new settings_tree())
{
	// copy property tree
	_settings = script._settings;

	// section tree is non-copyable
	_sections.clear();
}

axis::services::configuration::XmlConfigurationScript::XmlConfigurationScript( TiXmlNode *root, const axis::String& sectionName ) : _settings(*new settings_tree())
{
	_root = root;
	_sectionName = sectionName;
}

axis::services::configuration::XmlConfigurationScript::~XmlConfigurationScript( void )
{
	wipe_sections();
	delete &_settings;
}

void axis::services::configuration::XmlConfigurationScript::wipe_sections( void )
{
	// wipe out our dependent objects
	for (section_tree::iterator it = _sections.begin(); it != _sections.end(); ++it)
	{
		delete it->second;
	}
	_sections.clear();
}

axis::services::configuration::XmlConfigurationScript& axis::services::configuration::XmlConfigurationScript::ReadFromFile( const axis::String& fileName )
{
	XmlConfigurationScript& s = *new XmlConfigurationScript();
	settings_tree &settings = s._settings;

	// try to open file if it exists
	input_stream file;
	file.open(fileName.data(), input_stream::in);
	file.close();
	if (file.fail())
	{	// failed to open file
		throw axis::foundation::IOException(_T("Insufficient access permissions or file doen't exist."));
	}

	// parse XML
	TiXmlNode *rootNode = NULL;
	try
	{
		// set load parameters
		TiXmlBase::SetCondenseWhiteSpace(true);

		// try to load data		
		if (!settings.LoadFile(fileName.data()))
		{	// load error; check if it is due to empty document
			if (settings.ErrorId() == TIXML_ERROR_DOCUMENT_EMPTY)
			{	// yes, it is
				throw axis::foundation::ConfigurationNotFoundException(_T("Unexpected: empty file."));
			}
			// failed for any other reason; probably due to malformed syntax
			throw 1;
		}

		// look for the root node
		rootNode = settings.FirstChild();
		while (rootNode->Type() != TiXmlNode::TINYXML_ELEMENT)
		{
			rootNode = rootNode->NextSibling();
			if (rootNode == NULL)
			{
				// cannot find configuration file root node
				throw axis::foundation::ConfigurationNotFoundException(_T("Unexpected: root node not found."));
			}
		}		
	}
	catch(axis::foundation::ConfigurationNotFoundException e)
	{	// rethrow
		throw e;
	}
	catch (...)
	{	
		// free resources
		delete &s;
		throw axis::foundation::SyntaxMismatchException(_T("Malformed XML syntax in file '") + fileName + _T("'."));
	}

	s._root = rootNode;
	s._sectionName = s._root->Value();
	return s;
}

axis::String axis::services::configuration::XmlConfigurationScript::GetAttributeValue( const axis::String attributeName )
{
	if(_root->ToElement()->Attribute(attributeName) == NULL)
	{
		throw axis::foundation::ElementNotFoundException();
	}
	return axis::String(_root->ToElement()->Attribute(attributeName));
}

axis::services::configuration::ConfigurationScript& axis::services::configuration::XmlConfigurationScript::GetSection( const axis::String& sectionName )
{
	if (!ContainsSection(sectionName))
	{
		throw axis::foundation::ElementNotFoundException();
	}

	// retrieve node
	TiXmlNode *node = _root->FirstChildElement(sectionName);

	// create instance
	return create_dependent_instance(node, sectionName);
}

bool axis::services::configuration::XmlConfigurationScript::HasMoreSiblingsSection( void )
{
	// if this is the topmost level, obviously we don't have any siblings
	if (_root->Parent() == NULL)
	{
		return false;
	}

	// search for a sibling and ensure it is a XML element
	TiXmlNode *parent = _root->Parent();
	TiXmlNode *node = _root;
	do 
	{
		node = parent->IterateChildren(node);
		if (node != NULL)
		{
			if (node->ToElement() != NULL) return true;	// valid sibling found
		}
	} while (node != NULL);

	return false;	// no more valid siblings found
}

axis::services::configuration::ConfigurationScript& axis::services::configuration::XmlConfigurationScript::GetNextSiblingSection( void )
{
	if (!HasMoreSiblingsSection())
	{
		throw axis::foundation::ElementNotFoundException();
	}

	// retrieve node
	TiXmlNode *parent = _root->Parent();
	TiXmlNode *node = _root;
	do 
	{
		node = parent->IterateChildren(node);
		if (node != NULL)
		{
			if (node->ToElement() != NULL) break;
		}
	} while (node != NULL);

	// create instance
	return create_dependent_instance(node, node->Value());
}

bool axis::services::configuration::XmlConfigurationScript::ContainsAttribute( const axis::String& attributeName )
{	
	return _root->ToElement()->Attribute(attributeName) != NULL;
}

axis::String axis::services::configuration::XmlConfigurationScript::GetAttributeWithDefault( const axis::String& attributeName, const axis::String& defaultValue )
{
	if(_root->ToElement()->Attribute(attributeName) == NULL)
	{
		return axis::String(defaultValue);
	}
	return axis::String(*_root->ToElement()->Attribute(attributeName));
}

axis::String axis::services::configuration::XmlConfigurationScript::GetConfigurationText( void )
{
	if(_root->ToElement()->GetText() == NULL)
	{
		return axis::String();
	}
	return axis::String(_root->ToElement()->GetText());
}

bool axis::services::configuration::XmlConfigurationScript::ContainsConfigurationText( void )
{
	return _root->ToElement()->GetText() != NULL;
}

bool axis::services::configuration::XmlConfigurationScript::ContainsSection( const axis::String& sectionName )
{
	return _root->FirstChild(sectionName) != NULL;
}

axis::services::configuration::ConfigurationScript& axis::services::configuration::XmlConfigurationScript::create_dependent_instance( TiXmlNode *root, const axis::String& sectionName )
{
	// return the same object if we have already created
	if (_sections.find(root) != _sections.end())
	{
		return *_sections[root];
	}

	// this is a new request; create new object
	XmlConfigurationScript& cs = *new XmlConfigurationScript(root, sectionName);
	_sections[root] = &cs;	// remember this object so we can destroy it later
	return cs;
}

axis::services::configuration::ConfigurationScript& axis::services::configuration::XmlConfigurationScript::GetFirstChildSection( void )
{
	if (!HasChildSections())
	{
		throw axis::foundation::ElementNotFoundException();
	}

	TiXmlNode *node = _root->FirstChildElement();
	return create_dependent_instance(node, node->Value());
}

bool axis::services::configuration::XmlConfigurationScript::HasChildSections( void )
{
	return _root->FirstChildElement() != NULL;
}

axis::String axis::services::configuration::XmlConfigurationScript::GetSectionName( void ) const
{
	return _sectionName;
}

axis::services::configuration::ConfigurationScript * axis::services::configuration::XmlConfigurationScript::FindNextSibling( const axis::String& sectionName )
{
	if (!HasMoreSiblingsSection())
	{	// there is no more siblings
		return NULL;
	}

	// retrieve next sibling
	TiXmlNode *parent = _root->Parent();
	TiXmlNode *node = _root;
	do 
	{
		node = parent->IterateChildren(sectionName, node);
		if (node->ToElement() != NULL)
		{	// next sibling found
			break;
		}
	} while (node != NULL);
	
	if (node == NULL) return NULL;	// sibling not found

	// create instance
	return &create_dependent_instance(node, sectionName);
}

axis::String axis::services::configuration::XmlConfigurationScript::GetConfigurationPath( void ) const
{
	TiXmlNode *node = _root->Parent();
	int level = get_node_order_number(_root);

	axis::String str = _sectionName;

	if (level > 0)
	{
		str.append(_T("<")).append(boost::lexical_cast<axis::String>(level)).append(_T(">"));		
	}
	while(node->ToElement() != NULL)
	{
		axis::String aux(node->ToElement()->Value());

		level = get_node_order_number(node);
		if (level > 0)
		{
			aux.append(_T("<")).append(boost::lexical_cast<axis::String>(level)).append(_T(">"));		
		}

		aux.append(_T("."));
		str.insert(str.begin(),aux.begin(), aux.end());

		node = node->Parent();
	}

	return str;
}

int axis::services::configuration::XmlConfigurationScript::get_node_order_number( TiXmlNode *node ) const
{
	int order = 0;
	TiXmlNode *parent = node->Parent();
	TiXmlNode *item = parent->FirstChildElement();

	if (axis::String(item->Value()).compare(node->Value()) != 0)
	{	// search for first node with the same name
		parent->IterateChildren(node->Value(), item);
	}

	// search if there are previous elements in this level with the same name and count them
	while(item != node)
	{
		item = parent->IterateChildren(node->Value(), item);
		order++;
	}

	if (order == 0)
	{	// there might be a possibility that we are the first of a series of items 
		// with the same name -- check it
		item = parent->IterateChildren(node->Value(), node);
		if (item != NULL)	// there are more of them
		{	// set our order number to 1 (the first one, that is)
			return 1;
		}
		return 0;
	}

	return ++order;
}

void axis::services::configuration::XmlConfigurationScript::Destroy( void ) const
{
	delete this;
}