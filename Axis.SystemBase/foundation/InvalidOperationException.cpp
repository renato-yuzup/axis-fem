#include "InvalidOperationException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidOperationException::InvalidOperationException(void) : AxisException()
{
	// no more ops here
}

InvalidOperationException::InvalidOperationException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

InvalidOperationException::InvalidOperationException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

InvalidOperationException::InvalidOperationException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::InvalidOperationException::InvalidOperationException( const InvalidOperationException& ex )
{
	Copy(ex);
}

InvalidOperationException& axis::foundation::InvalidOperationException::operator=( const InvalidOperationException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InvalidOperationException::Clone( void ) const
{
	return *new InvalidOperationException(*this);
}

axis::String axis::foundation::InvalidOperationException::GetTypeName( void ) const
{
	return _T("InvalidOperationException");
}