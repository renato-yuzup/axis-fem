
#include "IOException.hpp"

using namespace axis::foundation;
using namespace axis;

IOException::IOException(void) : AxisException()
{
	// no more ops here
}

IOException::IOException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

IOException::IOException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

IOException::IOException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::IOException::IOException( const IOException& ex )
{
	Copy(ex);
}

IOException& axis::foundation::IOException::operator=( const IOException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::IOException::Clone( void ) const
{
	return *new IOException(*this);
}

axis::String axis::foundation::IOException::GetTypeName( void ) const
{
	return _T("IOException");
}