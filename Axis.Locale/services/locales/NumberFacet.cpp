#include "stdafx.h"
#include "NumberFacet.hpp"
#include <sstream>
#include <assert.h>

#ifdef _UNICODE
	typedef std::wstringstream std_string_stream;
#else
	typedef std::stringstream std_string_stream;
#endif

axis::services::locales::NumberFacet::NumberFacet( void )
{
	// nothing to do here
}

axis::services::locales::NumberFacet::~NumberFacet( void )
{
	// nothing to do here
}

axis::String axis::services::locales::NumberFacet::DoToPercent( double percent, unsigned int numberOfDigits ) const
{
	std_string_stream stream;
	stream.setf(std::ios::floatfield);
	stream.precision(numberOfDigits);
	stream << (percent * 100.0);
	return axis::String(stream.str().data()) + _T("%");
}

axis::String axis::services::locales::NumberFacet::DoToString( double number ) const
{
	std_string_stream stream;
	stream.setf(std::ios::floatfield);
	stream << number;
	return axis::String(stream.str().data());
}

axis::String axis::services::locales::NumberFacet::DoToString( double number, unsigned int numberOfDigits ) const
{
	std_string_stream stream;
	stream.setf(std::ios::floatfield);
	stream.precision(numberOfDigits);
	stream << number;
	return axis::String(stream.str().data());
}

axis::String axis::services::locales::NumberFacet::DoToString( long number ) const
{
	return axis::String::int_parse(number);
}

axis::String axis::services::locales::NumberFacet::DoToString( unsigned long number ) const
{
	return axis::String::int_parse(number);
}

axis::String axis::services::locales::NumberFacet::DoToScientificNotation( double number ) const
{
	std_string_stream stream;
	stream.setf(std::ios::floatfield | std::ios::scientific);
	stream << number;
	return axis::String(stream.str().data());
}

axis::String axis::services::locales::NumberFacet::DoToScientificNotation( double number, int maxDecimalDigits ) const
{
	std_string_stream stream;
	stream.setf(std::ios::floatfield | std::ios::scientific);
	stream.precision(maxDecimalDigits);
	stream << number;
	return axis::String(stream.str().data());
}

axis::String axis::services::locales::NumberFacet::DoToEngineeringNotation( double number ) const
{
	if (number == 0.0)
	{
		return ToScientificNotation(0.0);
	}

	try
	{
		int numberExponent = (int)log10(abs(number));

		int integerExponent = numberExponent % 3;
		int decimalExponent = numberExponent - integerExponent;

		if (integerExponent < 0)
		{
			integerExponent += 3;
			decimalExponent -= 3;
		}

		double engNum = number / pow(10, numberExponent - decimalExponent);

		std_string_stream stream;
		stream.setf(std::ios::floatfield);
		stream << number;
		axis::String str = stream.str().data();
		str += (decimalExponent >= 0)? _T("E+") : _T("E");
		str += axis::String::int_parse((long)decimalExponent);

		return str;
	}
	catch (...)
	{
		return _T("#Err");
	}
}

axis::String axis::services::locales::NumberFacet::DoToEngineeringNotation( double number, int maxDecimalDigits ) const
{
	if (number == 0.0)
	{
		return ToScientificNotation(0.0);
	}

	try
	{
		int numberExponent = (int)log10(abs(number));

		int integerExponent = numberExponent % 3;
		int decimalExponent = numberExponent - integerExponent;

		if (integerExponent < 0)
		{
			integerExponent += 3;
			decimalExponent -= 3;
		}

		double engNum = number / pow(10, numberExponent - decimalExponent);

		std_string_stream stream;
		stream.setf(std::ios::floatfield);
		stream.precision(maxDecimalDigits);
		stream << number;
		axis::String str = stream.str().data();
		str += (decimalExponent >= 0)? _T("E+") : _T("E");
		str += axis::String::int_parse((long)decimalExponent);

		return str;
	}
	catch (...)
	{
		return _T("#Err");
	}
}

axis::String axis::services::locales::NumberFacet::ToPercent( double percent, unsigned int numberOfDigits /*= 0*/ ) const
{
	String s = DoToPercent(percent, numberOfDigits);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToString( double number ) const
{
	String s = DoToString(number);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToString( double number, unsigned int numberOfDigits ) const
{
	String s = DoToString(number, numberOfDigits);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToString( long number ) const
{
	String s = DoToString(number);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToString( unsigned long number ) const
{
	String s = DoToString(number);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToScientificNotation( double number ) const
{
	String s = DoToScientificNotation(number);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToScientificNotation( double number, int maxDecimalDigits ) const
{
	String s = DoToScientificNotation(number, maxDecimalDigits);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToEngineeringNotation( double number ) const
{
	String s = DoToEngineeringNotation(number);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}

axis::String axis::services::locales::NumberFacet::ToEngineeringNotation( double number, int maxDecimalDigits ) const
{
	String s = DoToEngineeringNotation(number, maxDecimalDigits);
	s.trim();
	assert(!s.empty() && "Invalid number formatting: result is empty.");
	return s;
}
