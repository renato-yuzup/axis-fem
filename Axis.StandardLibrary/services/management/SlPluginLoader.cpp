#include "StdAfx.h"
#include "SlPluginLoader.hpp"

using namespace axis::foundation;

axis::services::management::SlPluginLoader::~SlPluginLoader( void )
{
	// nothing to do here
}

void axis::services::management::SlPluginLoader::StartPlugin( GlobalProviderCatalog& manager )
{
	// meanwhile, this is a no-op function
}

void axis::services::management::SlPluginLoader::Destroy( void ) const
{
	delete this;
}

void axis::services::management::SlPluginLoader::UnloadPlugin( GlobalProviderCatalog& manager )
{
	// nothing to do at this time
}

axis::services::management::PluginInfo axis::services::management::SlPluginLoader::GetPluginInformation( void ) const
{
	return PluginInfo(_T("axis Standard Library"),
					  _T("Library containing the basic elements and materials types as well as basic numerical algorithms used in most analysis."),
					  _T("axis.StandardLibrary"),
					  _T("Renato T. Yamassaki"),
					  _T("(c) Copyright 2012"),
					  0, 1, 1, 1, 
					  _T(""));
}
