#include "StringValue.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::StringValue::StringValue( const axis::String& value ) :
_value(value)
{
	/* nothing to do */
}

aslse::StringValue::~StringValue( void )
{
	/* nothing to do */
}

bool aslse::StringValue::IsString( void ) const
{
	return true;
}

bool aslse::StringValue::IsId( void ) const
{
	return false;
}

bool aslse::StringValue::IsNumeric( void ) const
{
	return false;
}

axis::String aslse::StringValue::ToString( void ) const
{
	return _value;
}

aslse::ParameterValue& aslse::StringValue::Clone( void ) const
{
	return *new StringValue(_value);
}

axis::String aslse::StringValue::GetValue( void ) const
{
	return _value;
}
