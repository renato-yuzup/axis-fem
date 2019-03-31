#include "ArgumentException.hpp"

using namespace axis::foundation;
using namespace axis;

ArgumentException::ArgumentException(void) : AxisException()
{
	// intentionally left blank
}

ArgumentException::ArgumentException( const String& message, const AxisException * innerException ) : AxisException(message, innerException), _argumentName(_TEXT(""))
{
	// intentionally left blank
}

ArgumentException::ArgumentException( const AxisException * innerException ) : AxisException(innerException), _argumentName(_TEXT(""))
{
	// intentionally left blank
}

ArgumentException::ArgumentException( const String& argumentName ) : _argumentName(argumentName)
{
	// intentionally left blank
}

axis::foundation::ArgumentException::ArgumentException( const String& argumentName, String& message ) : AxisException(message), _argumentName(argumentName)
{
	// intentionally left blank
}

axis::foundation::ArgumentException::ArgumentException( const String& argumentName, String& message, const AxisException * innerException ) : AxisException(message, innerException), _argumentName(argumentName)
{
	// intentionally left blank
}

axis::foundation::ArgumentException::ArgumentException( const ArgumentException& ex ) : _argumentName(ex._argumentName)
{
	Copy(ex);
}

const String& axis::foundation::ArgumentException::GetArgumentName( void ) const
{
	return _argumentName;
}

ArgumentException& axis::foundation::ArgumentException::operator=( const ArgumentException& ex )
{
	Copy(ex);
	_argumentName = ex._argumentName;
	return *this;
}

AxisException& axis::foundation::ArgumentException::Clone( void ) const
{
	return *new ArgumentException(*this);
}

axis::String axis::foundation::ArgumentException::GetTypeName( void ) const
{
	return _T("ArgumentException");
}