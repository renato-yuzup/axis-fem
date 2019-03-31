#include "OverflowException.hpp"

using namespace axis::foundation;
using namespace axis;

OverflowException::OverflowException(void) : AxisException()
{
	// no more ops here
}

OverflowException::OverflowException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

OverflowException::OverflowException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

OverflowException::OverflowException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::OverflowException::OverflowException( const OverflowException& ex )
{
	Copy(ex);
}

OverflowException& axis::foundation::OverflowException::operator=( const OverflowException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::OverflowException::Clone( void ) const
{
	return *new OverflowException(*this);
}

axis::String axis::foundation::OverflowException::GetTypeName( void ) const
{
	return _T("OverflowException");
}