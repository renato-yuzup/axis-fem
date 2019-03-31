#include "NullValue.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::NullValue::NullValue( void )
{
	/* nothing to do here */
}

aslse::NullValue::~NullValue( void )
{
	/* nothing to do here */
}

bool aslse::NullValue::IsAssignment( void ) const
{
	return false;
}

bool aslse::NullValue::IsAtomic( void ) const
{
	return false;
}

bool aslse::NullValue::IsNull( void ) const
{
	return true;
}

bool aslse::NullValue::IsArray( void ) const
{
	return false;
}

axis::String aslse::NullValue::ToString( void ) const
{
	return axis::String();
}

aslse::ParameterValue& aslse::NullValue::Clone( void ) const
{
	return *new NullValue();
}
