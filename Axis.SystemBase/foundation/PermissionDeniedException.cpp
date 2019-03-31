#include "PermissionDeniedException.hpp"

using namespace axis::foundation;
using namespace axis;

PermissionDeniedException::PermissionDeniedException(void) : AxisException()
{
	// no more ops here
}

PermissionDeniedException::PermissionDeniedException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

PermissionDeniedException::PermissionDeniedException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

PermissionDeniedException::PermissionDeniedException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::PermissionDeniedException::PermissionDeniedException( const PermissionDeniedException& ex )
{
	Copy(ex);
}

PermissionDeniedException& axis::foundation::PermissionDeniedException::operator=( const PermissionDeniedException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::PermissionDeniedException::Clone( void ) const
{
	return *new PermissionDeniedException(*this);
}

axis::String axis::foundation::PermissionDeniedException::GetTypeName( void ) const
{
	return _T("PermissionDeniedException");
}