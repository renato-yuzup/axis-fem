#include "NotImplementedException.hpp"

using namespace axis::foundation;
using namespace axis;

NotImplementedException::NotImplementedException(void) : AxisException()
{
	// no more ops here
}

NotImplementedException::NotImplementedException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

NotImplementedException::NotImplementedException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

NotImplementedException::NotImplementedException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::NotImplementedException::NotImplementedException( const NotImplementedException& ex )
{
	Copy(ex);
}

axis::foundation::NotImplementedException axis::foundation::NotImplementedException::operator=( const NotImplementedException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::NotImplementedException::Clone( void ) const
{
	return *new NotImplementedException(*this);
}

axis::String axis::foundation::NotImplementedException::GetTypeName( void ) const
{
	return _T("NotImplementedException");
}