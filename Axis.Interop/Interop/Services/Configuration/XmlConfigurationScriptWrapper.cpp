#include "StdAfx.h"
#include "XmlConfigurationScriptWrapper.hpp"
#include "../../foundation/StringWrapService.hpp"
#include "services/configuration/ScriptFactory.hpp"

using namespace axis::Interop::foundation;

axis::Interop::Services::Configuration::XmlConfigurationScriptWrapper::XmlConfigurationScriptWrapper( System::String ^fileName )
{
	_fileName = gcnew System::String(fileName->ToCharArray());
	
	axis::String s = StringWrapService::WrapToAxisString(fileName);
	_xmlScript = &axis::services::configuration::ScriptFactory::ReadFromXmlFile(s);
}

System::String ^axis::Interop::Services::Configuration::XmlConfigurationScriptWrapper::GetFileName( void )
{
	return gcnew System::String(_fileName->ToCharArray());
}

axis::Interop::Services::Configuration::XmlConfigurationScriptWrapper::~XmlConfigurationScriptWrapper( void )
{
	// call finalizer to free unmanaged objects resources
	this->!XmlConfigurationScriptWrapper();
}

axis::Interop::Services::Configuration::XmlConfigurationScriptWrapper::!XmlConfigurationScriptWrapper( void )
{
	delete _xmlScript;
	_xmlScript = NULL;
}

axis::services::configuration::ConfigurationScript * axis::Interop::Services::Configuration::XmlConfigurationScriptWrapper::GetWrappedScript( void )
{
	return _xmlScript;
}