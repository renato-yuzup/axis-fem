#include "OutOfMemoryException.hpp"

using namespace axis::foundation;
using namespace axis;

OutOfMemoryException::OutOfMemoryException(void) : ApplicationErrorException()
{
	// no more ops here
}

OutOfMemoryException::OutOfMemoryException( const String& message, const AxisException * innerException ) : ApplicationErrorException(message, innerException)
{
	// no more ops here
}

OutOfMemoryException::OutOfMemoryException( const AxisException * innerException ) : ApplicationErrorException(innerException)
{
	// no more ops here
}

OutOfMemoryException::OutOfMemoryException( const String& message ) : ApplicationErrorException(message)
{
	// no more ops here
}

axis::foundation::OutOfMemoryException::OutOfMemoryException( const OutOfMemoryException& ex )
{
	Copy(ex);
}

OutOfMemoryException& axis::foundation::OutOfMemoryException::operator=( const OutOfMemoryException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::OutOfMemoryException::Clone( void ) const
{
	return *new OutOfMemoryException(*this);
}

axis::String axis::foundation::OutOfMemoryException::GetTypeName( void ) const
{
	return _T("OutOfMemoryException");
}