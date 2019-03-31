#include "stdafx.h"
#include "LocaleLocator.hpp"
#include "LocaleLocator_pimpl.hpp"
#include "SystemCollationFacet.hpp"
#include "SystemDateTimeFacet.hpp"
#include "SystemNumberFacet.hpp"

#include "services/io/FileSystem.hpp"
#include "services/io/DirectoryNavigator.hpp"

#include <assert.h>


using namespace axis::services::io;

axis::services::locales::LocaleLocator * axis::services::locales::LocaleLocator::_singleInstance = NULL;

axis::services::locales::LocaleLocator::LocaleLocator( void )
{
	_localeComponents = new LocaleDescriptor[2];
	_localeComponents[0].DateTimeLocale = new DateTimeFacet();
	_localeComponents[0].NumberLocale = new NumberFacet();
	_localeComponents[0].Collation = new CollationFacet();
	_localeComponents[0].TranslationLocale = new TranslationFacet();
	_localeComponents[0].RegionLocale = new Locale(_T("*"), 
												   *_localeComponents[0].DateTimeLocale,
												   *_localeComponents[0].NumberLocale,
												   *_localeComponents[0].Collation,
												   *_localeComponents[0].TranslationLocale
												   );

	_localeComponents[1].DateTimeLocale = new SystemDateTimeFacet();
	_localeComponents[1].NumberLocale = new SystemNumberFacet();
	_localeComponents[1].Collation = new SystemCollationFacet();
	_localeComponents[1].TranslationLocale = new TranslationFacet();
	_localeComponents[1].RegionLocale = new Locale(_T("*"), 
												   *_localeComponents[1].DateTimeLocale,
												   *_localeComponents[1].NumberLocale,
												   *_localeComponents[1].Collation,
												   *_localeComponents[1].TranslationLocale
												   );

	_localeData = new LocaleData(*_localeComponents[0].RegionLocale);
}

axis::services::locales::LocaleLocator::~LocaleLocator( void )
{
	delete _localeData;

	delete _localeComponents[0].Collation;
	delete _localeComponents[0].DateTimeLocale;
	delete _localeComponents[0].NumberLocale;
	delete _localeComponents[0].TranslationLocale;
	delete _localeComponents[0].RegionLocale;

	delete _localeComponents[1].Collation;
	delete _localeComponents[1].DateTimeLocale;
	delete _localeComponents[1].NumberLocale;
	delete _localeComponents[1].TranslationLocale;
	delete _localeComponents[1].RegionLocale;

	delete [] _localeComponents;

	_localeData = NULL;
	_localeComponents = NULL;
}

axis::services::locales::LocaleLocator& axis::services::locales::LocaleLocator::GetLocator( void )
{
	if (_singleInstance == NULL)
	{
		_singleInstance = new axis::services::locales::LocaleLocator();
	}
	return *_singleInstance;
}

const axis::services::locales::Locale& axis::services::locales::LocaleLocator::GetDefaultLocale( void ) const
{
	return *_localeComponents[0].RegionLocale;
}

const axis::services::locales::Locale& axis::services::locales::LocaleLocator::GetSystemRegionLocale( void ) const
{
	return *_localeComponents[1].RegionLocale;
}

const axis::services::locales::Locale& axis::services::locales::LocaleLocator::GetLocale( const axis::String& localeCode ) const
{
	if (localeCode == _T("*"))
	{
		return GetDefaultLocale();
	}
	else if (localeCode == _T("+"))
	{
		return GetSystemRegionLocale();
	}
	return _localeData->GetDictionary().GetLocale(localeCode);
}

size_type axis::services::locales::LocaleLocator::GetRegisteredLocaleCount( void ) const
{
	return 2 + _localeData->GetDictionary().GetLocaleCount();
}

bool axis::services::locales::LocaleLocator::IsRegistered( const axis::String& localeCode ) const
{
	if (localeCode == _T("*") || localeCode == _T("+"))
	{
		return true;
	}
	return _localeData->GetDictionary().ExistsLocale(localeCode);
}

void axis::services::locales::LocaleLocator::LoadLocales( void )
{
	// TODO: we have to implement external locale loading process!
	// _localeData->LoadApplicationLocales(FileSystem::GetLocaleFolder());
}

void axis::services::locales::LocaleLocator::UnloadLocales( void )
{
	// TODO: we have to implement external locale unloading process!
}

const axis::services::locales::Locale& axis::services::locales::LocaleLocator::GetGlobalLocale( void ) const
{
	return _localeData->GetGlobalLocale();
}

void axis::services::locales::LocaleLocator::SetGlobalLocale( const Locale& locale )
{
	_localeData->SetGlobalLocale(locale);
}
