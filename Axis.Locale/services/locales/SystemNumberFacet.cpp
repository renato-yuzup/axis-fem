#include "stdafx.h"
#include "SystemNumberFacet.hpp"
#include <boost/locale.hpp>

using namespace boost::locale;

#ifdef _UNICODE
typedef std::wstringstream string_stream;
#else
typedef std::stringstream string_stream;
#endif

axis::services::locales::SystemNumberFacet::SystemNumberFacet( void )
{
// 	generator gen;
// 	_systemLocale = gen("");
}

axis::services::locales::SystemNumberFacet::~SystemNumberFacet( void )
{
	// nothing to do here
}

axis::String axis::services::locales::SystemNumberFacet::DoToPercent( double percent, unsigned int numberOfDigits ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss.precision(numberOfDigits);
	ss << as::percent << percent;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemNumberFacet::DoToString( double number ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::number << number;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemNumberFacet::DoToString( double number, unsigned int numberOfDigits ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss.precision(numberOfDigits);
	ss << as::number << number;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemNumberFacet::DoToString( long number ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::number << number;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemNumberFacet::DoToString( unsigned long number ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::percent << number;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemNumberFacet::DoToScientificNotation( double number ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss.setf(std::ios_base::scientific);
	ss << as::number << number;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemNumberFacet::DoToScientificNotation( double number, int maxDecimalDigits ) const
{
	string_stream ss;
	ss.imbue(_systemLocale);
	ss.setf(std::ios_base::scientific);
	ss.precision(maxDecimalDigits);
	ss << as::number << number;
	return axis::String(ss.str().c_str());
}
