#include "OutOfBoundsException.hpp"

using namespace axis::foundation;
using namespace axis;

OutOfBoundsException::OutOfBoundsException(void) : AxisException()
{
	// no more ops here
}

OutOfBoundsException::OutOfBoundsException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

OutOfBoundsException::OutOfBoundsException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

OutOfBoundsException::OutOfBoundsException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::OutOfBoundsException::OutOfBoundsException( const OutOfBoundsException& ex )
{
	Copy(ex);
}

OutOfBoundsException& axis::foundation::OutOfBoundsException::operator=( const OutOfBoundsException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::OutOfBoundsException::Clone( void ) const
{
	return *new OutOfBoundsException(*this);
}

axis::String axis::foundation::OutOfBoundsException::GetTypeName( void ) const
{
	return _T("OutOfBoundsException");
}