#include "EOFException.hpp"

using namespace axis::foundation;
using namespace axis;

EOFException::EOFException(void) : AxisException()
{
	// no more ops here
}

EOFException::EOFException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

EOFException::EOFException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

EOFException::EOFException( const String& message ) : AxisException(message)
{
	// no more ops here
}


axis::foundation::EOFException::EOFException( const EOFException& ex )
{
	Copy(ex);
}

EOFException& axis::foundation::EOFException::operator=( const EOFException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::EOFException::Clone( void ) const
{
	return *new EOFException(*this);
}

axis::String axis::foundation::EOFException::GetTypeName( void ) const
{
	return _T("EOFException");
}