#include "stdafx.h"
#include "CollationFacet.hpp"
#include "boost/locale.hpp"
#include <assert.h>

using namespace boost::locale;

class axis::services::locales::CollationFacet::CollationData
{
public:
	std::locale CollationLocale;
};


axis::services::locales::CollationFacet::CollationFacet( void )
{
	_data = new CollationData();

// 	generator gen;
// 	gen.categories(collation_facet);
// 	_data->CollationLocale = gen("en_US.UTF-8");
}

axis::services::locales::CollationFacet::~CollationFacet( void )
{
	delete _data;
}

int axis::services::locales::CollationFacet::CompareIdentical( const axis::String& s1, const axis::String& s2 ) const
{
	return s1.compare(s2);
}

int axis::services::locales::CollationFacet::DoCompareStrict( const axis::String& s1, const axis::String& s2 ) const
{
	return std::use_facet<collator<axis::String::char_type>>(_data->CollationLocale)
		.compare(collator_base::quaternary, s1.c_str(), s2.c_str());
}

int axis::services::locales::CollationFacet::DoCompareIgnoreCase( const axis::String& s1, const axis::String& s2 ) const
{
	return std::use_facet<collator<axis::String::char_type>>(_data->CollationLocale)
					.compare(collator_base::secondary, s1.c_str(), s2.c_str());
}

int axis::services::locales::CollationFacet::DoCompareIgnoreCaseAndAccents( const axis::String& s1, const axis::String& s2 ) const
{
	return std::use_facet<collator<axis::String::char_type>>(_data->CollationLocale)
		.compare(collator_base::primary, s1.c_str(), s2.c_str());
}

axis::services::locales::CollationFacet::DefaultCollationType axis::services::locales::CollationFacet::GetDefaultCollation( void ) const
{
	return Identical;
}



int axis::services::locales::CollationFacet::CompareStrict( const axis::String& s1, const axis::String& s2 ) const
{
	return DoCompareStrict(s1, s2);
}

int axis::services::locales::CollationFacet::CompareIgnoreCase( const axis::String& s1, const axis::String& s2 ) const
{
	return DoCompareIgnoreCase(s1, s2);
}

int axis::services::locales::CollationFacet::CompareIgnoreCaseAndAccents( const axis::String& s1, const axis::String& s2 ) const
{
	return DoCompareIgnoreCaseAndAccents(s1, s2);
}

int axis::services::locales::CollationFacet::Compare( const axis::String& s1, const axis::String& s2 ) const
{
	switch (GetDefaultCollation())
	{
	case Identical:
		return CompareIdentical(s1, s2);
	case Strict:
		return CompareStrict(s1, s2);
	case IgnoreCase:
		return CompareIgnoreCase(s1, s2);
	case IgnoreCaseAndAccents:
		return CompareIgnoreCaseAndAccents(s1, s2);
	default:
		break;
	}
	assert(!"Code execution should never reach here!");
	return 0;
}


