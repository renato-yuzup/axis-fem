#include "ExternalPluginAgent.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/definitions/ConfigurationFileDescriptor.hpp"
#include "services/io/FileSystem.hpp"
#include "services/management/WindowsPluginLibrarian.hpp"
#include "services/management/PluginLink.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "foundation/InvalidPluginException.hpp"
#include "foundation/BadPluginException.hpp"

namespace aaa = axis::application::agents;
namespace asc = axis::services::configuration;
namespace asmg = axis::services::management;
namespace af = axis::foundation;
namespace afc = axis::foundation::collections;
namespace afdf = axis::foundation::definitions;
namespace asio = axis::services::io;
namespace asmm = axis::services::messaging;

aaa::ExternalPluginAgent::ExternalPluginAgent( void )
{
	_librarian = new asmg::WindowsPluginLibrarian();
	_pluginInfoList = &afc::ObjectList::Create();
	_pluginCount = 0;
	_script = NULL;
	_manager = NULL;
}

aaa::ExternalPluginAgent::~ExternalPluginAgent( void )
{
	_pluginInfoList->Destroy();
	_pluginInfoList = NULL;
  _librarian->UnloadPluginLayer(asmg::kPhase3_VolatilePhase, *_manager);
  _librarian->UnloadPluginLayer(asmg::kPhase2_NonVolatilePhase, *_manager);
  _librarian->UnloadPluginLayer(asmg::kPhase1_SystemCustomizationPhase, *_manager);
  _librarian->UnloadPluginLayer(asmg::kPhase0_SystemPhase, *_manager);
  _librarian->Destroy();
  _librarian = NULL;
}

void aaa::ExternalPluginAgent::SetUp( asc::ConfigurationScript& script,
															        asmg::GlobalProviderCatalog& manager)
{
	_script = &script;
	_manager = &manager;
}

void aaa::ExternalPluginAgent::LoadSystemCustomizerPlugins( void )
{
	if (_script == NULL)
	{
		throw af::InvalidOperationException(
      _T("Cannot load plugins because no configuration script was specified."));
	}
	String sectionName = afdf::ConfigurationFileDescriptor::CustomSystemPluginsSectionName;
	if (!_script->ContainsSection(sectionName))
	{
		return;
	}

	asc::ConfigurationScript& pluginsConfig = _script->GetSection(sectionName);
	RunPluginDiscovery(pluginsConfig, asmg::kPhase1_SystemCustomizationPhase, *_manager);
}

void aaa::ExternalPluginAgent::LoadNonVolatilePlugins( void )
{
	if (_script == NULL)
	{
		throw af::InvalidOperationException(
      _T("Cannot load plugins because no configuration script was specified."));
	}
	String sectionName = afdf::ConfigurationFileDescriptor::NonVolatilePluginsSectionName;
	if (!_script->ContainsSection(sectionName))
	{
		return;
	}

	asc::ConfigurationScript& pluginsConfig = _script->GetSection(sectionName);
	RunPluginDiscovery(pluginsConfig, asmg::kPhase2_NonVolatilePhase, *_manager);
}

void aaa::ExternalPluginAgent::LoadVolatilePlugins( void )
{
	if (_script == NULL)
	{
		throw af::InvalidOperationException(
      _T("Cannot load plugins because no configuration script was specified."));
	}
	String sectionName = afdf::ConfigurationFileDescriptor::VolatilePluginsSectionName;
	if (!_script->ContainsSection(sectionName))
	{
		return;
	}

	asc::ConfigurationScript& pluginsConfig = _script->GetSection(sectionName);
	RunPluginDiscovery(pluginsConfig, asmg::kPhase3_VolatilePhase, *_manager);
}

void aaa::ExternalPluginAgent::RunPluginDiscovery( asc::ConfigurationScript& script, 
                                                   asmg::PluginLayer targetLayer, 
                                                   asmg::GlobalProviderCatalog& manager )
{
	String pluginSubSection = script.GetSectionName();

	// check if we have any possible plugin declaration
	if (!script.HasChildSections())
	{	// it doesn't, exit
		return;
	}

	// check each child section if it is valid and interpret it when possible
	asc::ConfigurationScript *pluginDescriptor = &script.GetFirstChildSection(); 
	bool hasMoreChildren = true;
	do
	{
		if (IsValidPluginDescriptor(*pluginDescriptor))
		{	// it is valid, interpret it
			InterpretPluginDescriptor(*pluginDescriptor, targetLayer);
		}
		else
		{	// invalid section
			DispatchMessage(asmm::WarningMessage(0x200301,
				String(_T("Invalid plugin descriptor in sub-section %1: %2."))
            .replace(_T("%1"), pluginSubSection)
            .replace(_T("%1"), pluginDescriptor->GetSectionName()),
				_T("Malformed configuration file")));
		}
		hasMoreChildren = pluginDescriptor->HasMoreSiblingsSection();
		if (hasMoreChildren)
		{
			pluginDescriptor = &pluginDescriptor->GetNextSiblingSection();
		}
	} while (hasMoreChildren);

	// load plugin layer
	_librarian->LoadPluginLayer(targetLayer, manager);
}

bool aaa::ExternalPluginAgent::IsValidPluginDescriptor( asc::ConfigurationScript& pluginDescriptor ) const
{
	bool ok = false;
	if (pluginDescriptor.GetSectionName() == 
      afdf::ConfigurationFileDescriptor::PluginDescriptorTypeName)
	{	// it is a single plugin descriptor
		ok = pluginDescriptor.ContainsAttribute(
      afdf::ConfigurationFileDescriptor::PluginLocationAttributeName);
	}
	else if(pluginDescriptor.GetSectionName() == 
          afdf::ConfigurationFileDescriptor::PluginDirectoryDescriptorName)
	{	// it is a plugin directory descriptor, we have to iterate through it
		ok = pluginDescriptor.ContainsAttribute(
      afdf::ConfigurationFileDescriptor::PluginDirectoryLocationAttributeName);

		String value = pluginDescriptor.GetAttributeWithDefault(
        afdf::ConfigurationFileDescriptor::PluginDirectoryRecursiveSearchAttributeName, _T("no"));
		value.trim().to_lower_case();
		ok = ok && ((value == _T("yes")) || (value == _T("no")));
	}
	return ok;
}

void aaa::ExternalPluginAgent::InterpretPluginDescriptor( asc::ConfigurationScript& pluginDescriptor, 
                                                          asmg::PluginLayer targetLayer )
{
	if (pluginDescriptor.GetSectionName() == afdf::ConfigurationFileDescriptor::PluginDescriptorTypeName)
	{	// it is a single plugin descriptor
		LoadSinglePlugin(pluginDescriptor.GetAttributeValue(
      afdf::ConfigurationFileDescriptor::PluginLocationAttributeName), targetLayer);
	}
	else if(pluginDescriptor.GetSectionName() == 
          afdf::ConfigurationFileDescriptor::PluginDirectoryDescriptorName)
	{	// it is a plugin directory descriptor, we have to iterate through it

		bool recursive = false;
		String value = pluginDescriptor.GetAttributeWithDefault(
			afdf::ConfigurationFileDescriptor::PluginDirectoryRecursiveSearchAttributeName, 
			_T("no"));
		value.trim().to_lower_case();
		recursive = (value == _T("yes"));
		ScanPluginLibrary(pluginDescriptor.GetAttributeValue(
      afdf::ConfigurationFileDescriptor::PluginDirectoryLocationAttributeName), targetLayer, recursive);
	}
}

void aaa::ExternalPluginAgent::LoadSinglePlugin( const axis::String& pluginPath, 
                                                 asmg::PluginLayer targetLayer )
{
	String s = pluginPath;
	s.trim();

	// transform into absolute path, if necessary
	if (!asio::FileSystem::IsAbsolutePath(s))
	{	
		s = asio::FileSystem::ConcatenatePath(asio::FileSystem::GetApplicationFolder(), s);
		s = asio::FileSystem::ToCanonicalPath(s);
	}
	if (!asio::FileSystem::ExistsFile(s))
	{	// file does not exist; ignore
		DispatchMessage(asmm::WarningMessage(0x200305, 
        _T("One plugin was not loaded because it is not present in the specified path: '%1'."), s));
		return;
	}

	try
	{
		const asmg::PluginConnector& c =_librarian->AddConnector(s, targetLayer);
    DispatchMessage(asmm::InfoMessage(0x10030A, _T("Plugin loaded: '") + c.GetFileName() + _T("'.")));
		_pluginCount++;
	}
	catch (InvalidPluginException&)
	{	// file is not a valid plugin library
		DispatchMessage(asmm::WarningMessage(0x200306,
			String(_T("Could not load the following plugin because it is invalid or corrupt: '%1'."))
      .replace(_T("%1"), s), String(_T("Invalid plugin"))));
	}
	catch (BadPluginException&)
	{	// even though the file is a plugin, it behaves strangely
		DispatchMessage(asmm::WarningMessage(0x200307, String(
      _T("File '%1' seems to be a plugin, but it generated an error when we were trying to load it."))
      .replace(_T("%1"), s), String(_T("Unrecognied plugin"))));
	}
}

void aaa::ExternalPluginAgent::ScanPluginLibrary( const axis::String& pluginFolderLocation, 
                                                  asmg::PluginLayer targetLayer, bool recursive )
{
	String s = pluginFolderLocation;
	s.trim();
	// transform path to absolute (if needed)
	if (!asio::FileSystem::IsAbsolutePath(s))
	{
    s = asio::FileSystem::ConcatenatePath(asio::FileSystem::GetApplicationFolder(), s);
    s = asio::FileSystem::ToCanonicalPath(s);
	}
	if (!asio::FileSystem::ExistsFile(s))
	{
		DispatchMessage(asmm::WarningMessage(0x200308, String(
      _T("There is no plugin library at the specified path because it doesn't exist. Supplied path is '%1'."))
      .replace(_T("%1"), s), _T("Plugin library missing")));
		return;
	}
	// check if the specified path is a directory
	if (!asio::FileSystem::IsDirectory(s))
	{
		DispatchMessage(asmm::WarningMessage(0x200309, String(_T(
      "Cannot use the specified plugin library: it does not point to a directory. Supplied path is '%1'."))
      .replace(_T("%1"), s), _T("Plugin library missing")));
		return;
	}
	// ok, start scanning directory tree
	asio::DirectoryNavigator& navigator = asio::DirectoryNavigator::Create(s, recursive);
	for (;navigator.HasNext(); navigator.GoNext())
	{
		if (IsValidPluginFile(navigator))
		{
			LoadSinglePlugin(navigator.GetFileName(), targetLayer);
		}
	}
	navigator.Destroy();
}

size_type aaa::ExternalPluginAgent::GetPluginCount( void ) const
{
	return _pluginCount;	
}

bool aaa::ExternalPluginAgent::IsValidPluginFile( const asio::DirectoryNavigator& file ) const
{
	String s = file.GetFileExtension();
	s.to_lower_case();
	return (s == _T("dll"));
}

void aaa::ExternalPluginAgent::EnumerateLoadedPlugins( void )
{
	for (asmg::PluginLibrarian::Iterator it = _librarian->GetIterator(); it.HasNext(); it.GoNext())
	{
		if (it->IsPluginReady() && it->IsPluginLoaded())
		{
			asmg::PluginInfo info = it->GetPluginInformation();
			String pluginPath = it->GetFileName();
			_pluginInfoList->Add(*new asmg::PluginLink(info, pluginPath));
		}
	}
}

asmg::PluginLink aaa::ExternalPluginAgent::GetPluginLinkInfo( size_type index ) const
{
	return static_cast<asmg::PluginLink&>(_pluginInfoList->Get(index));
}
