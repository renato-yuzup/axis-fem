#include "ApplicationErrorException.hpp"

using namespace axis::foundation;
using namespace axis;

ApplicationErrorException::ApplicationErrorException(void) : AxisException()
{
	// no more ops here
}

ApplicationErrorException::ApplicationErrorException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

ApplicationErrorException::ApplicationErrorException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

ApplicationErrorException::ApplicationErrorException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::ApplicationErrorException::ApplicationErrorException( const ApplicationErrorException& ex )
{
	Copy(ex);
}

ApplicationErrorException& axis::foundation::ApplicationErrorException::operator=( const ApplicationErrorException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::ApplicationErrorException::Clone( void ) const
{
	return *new ApplicationErrorException(*this);
}

axis::String axis::foundation::ApplicationErrorException::GetTypeName( void ) const
{
	return _T("ApplicationErrorException");
}