#include "stdafx.h"
#include "Locale.hpp"


axis::services::locales::Locale::Locale( const axis::String& localeCode, 
										 const DateTimeFacet& dateTimeFacet, 
										 const NumberFacet& numberFacet, 
										 const CollationFacet& stringFacet, 
										 const TranslationFacet& messageFacet ) :
_localeCode(localeCode), _dateTimeFacet(&dateTimeFacet), _numberFacet(&numberFacet), _stringFacet(&stringFacet), _messageFacet(&messageFacet)
{
	// nothing to do here
}

axis::services::locales::Locale::Locale( const Locale& other ) :
_localeCode(other.GetLocaleCode()), _dateTimeFacet(&other.GetDataTimeLocale()), _numberFacet(&other.GetNumberLocale()), _stringFacet(&other.GetCollation()), _messageFacet(&other.GetDictionary())
{
	// nothing to do here
}

axis::services::locales::Locale::~Locale( void )
{
	// nothing to do here
}

axis::services::locales::Locale& axis::services::locales::Locale::operator=( const Locale& other )
{
	_localeCode = other.GetLocaleCode();
	_dateTimeFacet = &other.GetDataTimeLocale();
	_numberFacet = &other.GetNumberLocale();
	_stringFacet = &other.GetCollation();
	_messageFacet = &other.GetDictionary();

	return *this;
}

const axis::services::locales::DateTimeFacet& axis::services::locales::Locale::GetDataTimeLocale( void ) const
{
	return *_dateTimeFacet;
}

const axis::services::locales::NumberFacet& axis::services::locales::Locale::GetNumberLocale( void ) const
{
	return *_numberFacet;
}

const axis::services::locales::CollationFacet& axis::services::locales::Locale::GetCollation( void ) const
{
	return *_stringFacet;
}

const axis::services::locales::TranslationFacet& axis::services::locales::Locale::GetDictionary( void ) const
{
	return *_messageFacet;
}

axis::String axis::services::locales::Locale::GetLocaleCode( void ) const
{
	return _localeCode;
}

bool axis::services::locales::Locale::operator==( const Locale& other ) const
{
	return _localeCode == other.GetLocaleCode() &&
		   _dateTimeFacet == &other.GetDataTimeLocale() &&
		   _messageFacet == &other.GetDictionary() &&
		   _numberFacet == &other.GetNumberLocale() &&
		   _stringFacet == &other.GetCollation();
}

bool axis::services::locales::Locale::operator!=( const Locale& other ) const
{
	return !(*this == other);
}
