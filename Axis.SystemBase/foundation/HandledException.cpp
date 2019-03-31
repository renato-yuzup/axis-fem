#include "HandledException.hpp"

using namespace axis::foundation;
using namespace axis;

HandledException::HandledException(void) : AxisException()
{
	// no more ops here
}

HandledException::HandledException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

HandledException::HandledException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

HandledException::HandledException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::HandledException::HandledException( const HandledException& ex )
{
	Copy(ex);
}

HandledException& axis::foundation::HandledException::operator=( const HandledException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::HandledException::Clone( void ) const
{
	return *new HandledException(*this);
}

axis::String axis::foundation::HandledException::GetTypeName( void ) const
{
	return _T("HandledException");
}