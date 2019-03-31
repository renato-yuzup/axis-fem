#pragma once
#include <map>
#include <set>
#include "LocaleLocator.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/ElementNotFoundException.hpp"

class axis::services::locales::LocaleLocator::LocaleData
{
public:
	class LocaleDictionary;
private:
	LocaleDictionary   *_dictionary;
	const Locale *_currentLocale;

	// disallow copy constructor and copy assignment
	LocaleData(const LocaleData& other);
	LocaleData& operator =(const LocaleData& other);

public:
	LocaleData(const Locale& currentLocale);
	~LocaleData(void);

	LocaleDictionary& GetDictionary(void) const;

	void LoadApplicationLocales( const axis::String& localeLibraryPath );

	const Locale& GetGlobalLocale(void) const;
	void SetGlobalLocale(const Locale& newLocale);
};

class axis::services::locales::LocaleLocator::LocaleData::LocaleDictionary
{
private:
	typedef std::map<axis::String, axis::services::locales::Locale *> locale_map;
	locale_map _locales;
public:
	void AddLocale(const axis::String& entryCode, axis::services::locales::Locale& locale);
	bool ExistsLocale(const axis::String& entryCode) const;
	axis::services::locales::Locale& GetLocale(const axis::String& entryCode) const;
	void RemoveLocale(const axis::String& entryCode);
	void UnloadLocales(void);

	size_type GetLocaleCount(void) const;
};

class axis::services::locales::LocaleLocator::LocaleDescriptor
{
public:
	CollationFacet   *Collation;
	DateTimeFacet    *DateTimeLocale;
	NumberFacet      *NumberLocale;
	TranslationFacet *TranslationLocale;
	Locale			 *RegionLocale;
};

