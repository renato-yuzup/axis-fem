#include "ElementNotFoundException.hpp"

using namespace axis::foundation;
using namespace axis;

ElementNotFoundException::ElementNotFoundException(void) : AxisException()
{
	// no more ops here
}

ElementNotFoundException::ElementNotFoundException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

ElementNotFoundException::ElementNotFoundException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

ElementNotFoundException::ElementNotFoundException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::ElementNotFoundException::ElementNotFoundException( const ElementNotFoundException& ex )
{
	Copy(ex);
}

ElementNotFoundException& axis::foundation::ElementNotFoundException::operator=( const ElementNotFoundException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::ElementNotFoundException::Clone( void ) const
{
	return *new ElementNotFoundException(*this);
}

axis::String axis::foundation::ElementNotFoundException::GetTypeName( void ) const
{
	return _T("ElementNotFoundException");
}