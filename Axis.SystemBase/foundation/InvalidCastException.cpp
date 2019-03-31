#include "InvalidCastException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidCastException::InvalidCastException(void) : AxisException()
{
	// no more ops here
}

InvalidCastException::InvalidCastException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

InvalidCastException::InvalidCastException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

InvalidCastException::InvalidCastException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::InvalidCastException::InvalidCastException( const InvalidCastException& ex )
{
	Copy(ex);
}

InvalidCastException& axis::foundation::InvalidCastException::operator=( const InvalidCastException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InvalidCastException::Clone( void ) const
{
	return *new InvalidCastException(*this);
}

axis::String axis::foundation::InvalidCastException::GetTypeName( void ) const
{
	return _T("InvalidCastException");
}