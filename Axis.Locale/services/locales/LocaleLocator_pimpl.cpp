#include "stdafx.h"
#include "LocaleLocator_pimpl.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/NotImplementedException.hpp"




/*
 *************************************************** LocaleData members ***************************************************
*/
axis::services::locales::LocaleLocator::LocaleData::LocaleData( const Locale& currentLocale ) :
_currentLocale(&currentLocale)
{
	_dictionary = new LocaleDictionary();
}

axis::services::locales::LocaleLocator::LocaleData::~LocaleData( void )
{
	delete _dictionary;
	_dictionary = NULL;
}

axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary& axis::services::locales::LocaleLocator::LocaleData::GetDictionary( void ) const
{
	return *_dictionary;
}

void axis::services::locales::LocaleLocator::LocaleData::LoadApplicationLocales( const axis::String& localeLibraryPath )
{
	throw axis::foundation::NotImplementedException();
}

const axis::services::locales::Locale& axis::services::locales::LocaleLocator::LocaleData::GetGlobalLocale( void ) const
{
	return *_currentLocale;
}

void axis::services::locales::LocaleLocator::LocaleData::SetGlobalLocale( const Locale& newLocale )
{
	_currentLocale = &newLocale;
}




/*
 ************************************************ LocaleDictionary members ************************************************
*/
void axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary::AddLocale( const axis::String& entryCode, axis::services::locales::Locale& locale )
{
	if (ExistsLocale(entryCode))
	{
		throw axis::foundation::ArgumentException(_T("entryCode"));
	}
	_locales[entryCode] = &locale;
}

bool axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary::ExistsLocale( const axis::String& entryCode ) const
{
	return _locales.find(entryCode) != _locales.end();
}

axis::services::locales::Locale& axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary::GetLocale( const axis::String& entryCode ) const
{
	locale_map::const_iterator it = _locales.find(entryCode);
	if (it == _locales.end())
	{
		throw axis::foundation::ElementNotFoundException(_T("entryCode"));
	}
	return *it->second;
}

void axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary::RemoveLocale( const axis::String& entryCode )
{
	if (!ExistsLocale(entryCode))
	{
		throw axis::foundation::ArgumentException(_T("entryCode"));
	}
	locale_map::iterator it = _locales.find(entryCode);
	Locale *locale = (*it).second;
	_locales.erase(entryCode);

	delete locale;
}

void axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary::UnloadLocales( void )
{
	locale_map::iterator end = _locales.end();
	for (locale_map::iterator it = _locales.begin(); it != end; ++it)
	{
		Locale *locale = (*it).second;
		delete locale;
	}
	_locales.clear();
}

size_type axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary::GetLocaleCount( void ) const
{
	return (size_type)_locales.size();
}



