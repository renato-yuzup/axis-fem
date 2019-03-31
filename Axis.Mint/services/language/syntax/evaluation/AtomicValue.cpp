#include "AtomicValue.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::AtomicValue::~AtomicValue( void )
{
	/* nothing to do */
}

bool aslse::AtomicValue::IsAssignment( void ) const
{
	return false;
}

bool aslse::AtomicValue::IsAtomic( void ) const
{
	return true;
}

bool aslse::AtomicValue::IsNull( void ) const
{
	return false;
}

bool aslse::AtomicValue::IsArray( void ) const
{
	return false;
}
