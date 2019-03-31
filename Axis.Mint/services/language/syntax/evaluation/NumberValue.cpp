#include "NumberValue.hpp"
#include <boost/lexical_cast.hpp>
#include "string_traits.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::NumberValue::NumberValue( long value ) :
_longValue(value), _doubleValue((double)value), _isInteger(true)
{
	/* nothing to do here */
}

aslse::NumberValue::NumberValue( double value ) :
_longValue((long)value), _doubleValue(value), _isInteger(false)
{
	/* nothing to do here */
}

aslse::NumberValue::~NumberValue( void )
{
	/* nothing to do here */
}

bool aslse::NumberValue::IsString( void ) const
{
	return false;
}

bool aslse::NumberValue::IsId( void ) const
{
	return false;
}

bool aslse::NumberValue::IsNumeric( void ) const
{
	return true;
}

bool aslse::NumberValue::IsInteger( void ) const
{
	return _isInteger;
}

long aslse::NumberValue::GetLong( void ) const
{
	return _longValue;
}

double aslse::NumberValue::GetDouble( void ) const
{
	return _doubleValue;
}

axis::String aslse::NumberValue::ToString( void ) const
{
	axis::String str;
	if (_isInteger)
	{
		str = boost::lexical_cast<axis::String>(_longValue);
	}
	else
	{
		str = boost::lexical_cast<axis::String>(_doubleValue);	
	}
	return str;
}

aslse::ParameterValue& aslse::NumberValue::Clone( void ) const
{
	if (_isInteger)
	{
		return *new NumberValue(_longValue);
	}
	else
	{
		return *new NumberValue(_doubleValue);
	}
}
