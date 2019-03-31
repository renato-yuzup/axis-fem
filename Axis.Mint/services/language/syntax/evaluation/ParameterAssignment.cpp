#include "ParameterAssignment.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::ParameterAssignment::ParameterAssignment( const axis::String& idName, 
                                                 const ParameterValue& value ) :
_idName(idName), _value(value)
{
	/* nothing to do here */
}

aslse::ParameterAssignment::~ParameterAssignment( void )
{
	delete &_value;
}

bool aslse::ParameterAssignment::IsAssignment( void ) const
{
	return true;
}

bool aslse::ParameterAssignment::IsAtomic( void ) const
{
	return false;
}

bool aslse::ParameterAssignment::IsNull( void ) const
{
	return false;
}

bool aslse::ParameterAssignment::IsArray( void ) const
{
	return false;
}

axis::String aslse::ParameterAssignment::GetIdName( void ) const
{
	return _idName;
}

const aslse::ParameterValue& aslse::ParameterAssignment::GetRhsValue( void ) const
{
	return _value;
}

axis::String aslse::ParameterAssignment::ToString( void ) const
{
	axis::String str = _idName;
	str.append(_T("=")).append(_value.ToString());
	return str;
}

aslse::ParameterValue& aslse::ParameterAssignment::Clone( void ) const
{
	return *new ParameterAssignment(_idName, _value.Clone());
}
