#include "BadPluginException.hpp"

using namespace axis::foundation;
using namespace axis;

BadPluginException::BadPluginException(void) : AxisException()
{
	// no more ops here
}

BadPluginException::BadPluginException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

BadPluginException::BadPluginException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

BadPluginException::BadPluginException( const String& message ) : AxisException(message)
{
	// no more ops here
}

axis::foundation::BadPluginException::BadPluginException( const BadPluginException& ex )
{
	Copy(ex);
}

BadPluginException& axis::foundation::BadPluginException::operator=( const BadPluginException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::BadPluginException::Clone( void ) const
{
	return *new BadPluginException(*this);
}

axis::String axis::foundation::BadPluginException::GetTypeName( void ) const
{
	return _T("BadPluginException");
}