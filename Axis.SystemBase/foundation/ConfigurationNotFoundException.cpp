#include "ConfigurationNotFoundException.hpp"

using namespace axis::foundation;
using namespace axis;

ConfigurationNotFoundException::ConfigurationNotFoundException(void) : AxisException()
{
	// no more ops here

}

ConfigurationNotFoundException::ConfigurationNotFoundException(String &location) : AxisException()
{
	_path = location;
}

ConfigurationNotFoundException::ConfigurationNotFoundException(const char_type *location) : AxisException()
{
	_path = location;
}

ConfigurationNotFoundException::ConfigurationNotFoundException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

ConfigurationNotFoundException::ConfigurationNotFoundException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

axis::String axis::foundation::ConfigurationNotFoundException::GetPath( void )
{
	return _path;
}

void axis::foundation::ConfigurationNotFoundException::SetPath( const String & s )
{
	_path = s;
}

axis::foundation::ConfigurationNotFoundException::ConfigurationNotFoundException( const ConfigurationNotFoundException& ex )
{
	Copy(ex);
	_path = ex._path;
}

ConfigurationNotFoundException& axis::foundation::ConfigurationNotFoundException::operator=( const ConfigurationNotFoundException& ex )
{
	Copy(ex);
	_path = ex._path;
	return *this;
}

AxisException& axis::foundation::ConfigurationNotFoundException::Clone( void ) const
{
	return *new ConfigurationNotFoundException(*this);
}

axis::String axis::foundation::ConfigurationNotFoundException::GetTypeName( void ) const
{
	return _T("ConfigurationNotFoundException");
}