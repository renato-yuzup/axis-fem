#include "InvalidPluginException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidPluginException::InvalidPluginException(void) : AxisException()
{
	// no more ops here
}

InvalidPluginException::InvalidPluginException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

InvalidPluginException::InvalidPluginException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

InvalidPluginException::InvalidPluginException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::InvalidPluginException::InvalidPluginException( const InvalidPluginException& ex )
{
	Copy(ex);
}

InvalidPluginException& axis::foundation::InvalidPluginException::operator=( const InvalidPluginException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InvalidPluginException::Clone( void ) const
{
	return *new InvalidPluginException(*this);
}

axis::String axis::foundation::InvalidPluginException::GetTypeName( void ) const
{
	return _T("InvalidPluginException");
}