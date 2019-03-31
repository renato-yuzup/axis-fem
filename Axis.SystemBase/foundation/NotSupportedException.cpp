#include "NotSupportedException.hpp"

using namespace axis::foundation;
using namespace axis;

NotSupportedException::NotSupportedException(void) : AxisException()
{
	// no more ops here
}

NotSupportedException::NotSupportedException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

NotSupportedException::NotSupportedException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

NotSupportedException::NotSupportedException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::NotSupportedException::NotSupportedException( const NotSupportedException& ex )
{
	Copy(ex);
}

axis::foundation::NotSupportedException& axis::foundation::NotSupportedException::operator =(const NotSupportedException& ex)
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::NotSupportedException::Clone( void ) const
{
	return *new NotSupportedException(*this);
}

axis::String axis::foundation::NotSupportedException::GetTypeName( void ) const
{
	return _T("NotSupportedException");
}