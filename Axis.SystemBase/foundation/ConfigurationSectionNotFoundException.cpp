#include "ConfigurationSectionNotFoundException.hpp"

using namespace axis::foundation;
using namespace axis;

ConfigurationSectionNotFoundException::ConfigurationSectionNotFoundException(void) : AxisException()
{
	// no more ops here

}

ConfigurationSectionNotFoundException::ConfigurationSectionNotFoundException(String &location) : AxisException()
{
	_path = location;
}

ConfigurationSectionNotFoundException::ConfigurationSectionNotFoundException(const char_type *location) : AxisException()
{
	_path = location;
}

ConfigurationSectionNotFoundException::ConfigurationSectionNotFoundException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

ConfigurationSectionNotFoundException::ConfigurationSectionNotFoundException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

axis::String axis::foundation::ConfigurationSectionNotFoundException::GetPath( void )
{
	return _path;
}

void axis::foundation::ConfigurationSectionNotFoundException::SetPath( const String & s )
{
	_path = s;
}

axis::foundation::ConfigurationSectionNotFoundException::ConfigurationSectionNotFoundException( const ConfigurationSectionNotFoundException& ex )
{
	Copy(ex);
	_path = ex._path;
}

ConfigurationSectionNotFoundException& axis::foundation::ConfigurationSectionNotFoundException::operator=( const ConfigurationSectionNotFoundException& ex )
{
	Copy(ex);
	_path = ex._path;
	return *this;
}

AxisException& axis::foundation::ConfigurationSectionNotFoundException::Clone( void ) const
{
	return *new ConfigurationSectionNotFoundException(*this);
}

axis::String axis::foundation::ConfigurationSectionNotFoundException::GetTypeName( void ) const
{
	return _T("ConfigurationSectionNotFoundException");
}