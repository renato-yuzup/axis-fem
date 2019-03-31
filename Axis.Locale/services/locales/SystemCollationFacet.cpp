#include "stdafx.h"
#include "SystemCollationFacet.hpp"
#include "AxisString.hpp"
#include <boost/locale.hpp>

using namespace boost::locale;

axis::services::locales::SystemCollationFacet::SystemCollationFacet( void )
{
	// obtain current system locale
// 	generator gen;
// 	_systemLocale = gen("");
}

int axis::services::locales::SystemCollationFacet::DoCompareStrict( const axis::String& s1, const axis::String& s2 ) const
{
	return std::use_facet<collator<axis::String::char_type>>(_systemLocale).compare(collator_base::quaternary, s1.c_str(), s2.c_str());
}

int axis::services::locales::SystemCollationFacet::DoCompareIgnoreCase( const axis::String& s1, const axis::String& s2 ) const
{
	return std::use_facet<collator<axis::String::char_type>>(_systemLocale).compare(collator_base::secondary, s1.c_str(), s2.c_str());
}

int axis::services::locales::SystemCollationFacet::DoCompareIgnoreCaseAndAccents( const axis::String& s1, const axis::String& s2 ) const
{
	return std::use_facet<collator<axis::String::char_type>>(_systemLocale).compare(collator_base::primary, s1.c_str(), s2.c_str());
}

axis::services::locales::CollationFacet::DefaultCollationType axis::services::locales::SystemCollationFacet::GetDefaultCollation( void ) const
{
	return Strict;
}
