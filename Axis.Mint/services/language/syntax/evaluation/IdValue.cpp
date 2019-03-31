#include "IdValue.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::IdValue::IdValue( const axis::String& value ) :
_value(value)
{
	/* nothing to do here */
}

aslse::IdValue::~IdValue( void )
{
	/* nothing to do here */
}

bool aslse::IdValue::IsString( void ) const
{
	return false;
}

bool aslse::IdValue::IsId( void ) const
{
	return true;
}

bool aslse::IdValue::IsNumeric( void ) const
{
	return false;
}

axis::String aslse::IdValue::GetValue( void ) const
{
	return _value;
}

axis::String aslse::IdValue::ToString( void ) const
{
	return _value;
}

aslse::ParameterValue& aslse::IdValue::Clone( void ) const
{
	return *new IdValue(_value);
}