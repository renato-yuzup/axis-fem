#include "OutOfRangeException.hpp"

using namespace axis::foundation;
using namespace axis;

OutOfRangeException::OutOfRangeException(void) : AxisException()
{
	// no more ops here
}

OutOfRangeException::OutOfRangeException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

OutOfRangeException::OutOfRangeException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

OutOfRangeException::OutOfRangeException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::OutOfRangeException::OutOfRangeException( const OutOfRangeException& ex )
{
	Copy(ex);
}

OutOfRangeException& axis::foundation::OutOfRangeException::operator=( const OutOfRangeException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::OutOfRangeException::Clone( void ) const
{
	return *new OutOfRangeException(*this);
}

axis::String axis::foundation::OutOfRangeException::GetTypeName( void ) const
{
	return _T("OutOfRangeException");
}