#include "InvalidSyntaxException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidSyntaxException::InvalidSyntaxException(void) : AxisParserException()
{
	// no more ops here
}

InvalidSyntaxException::InvalidSyntaxException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

InvalidSyntaxException::InvalidSyntaxException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

InvalidSyntaxException::InvalidSyntaxException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}


InvalidSyntaxException::InvalidSyntaxException( const String& filename, unsigned long lineNumber ) : AxisParserException(filename, lineNumber)
{
	// no more ops here
}
InvalidSyntaxException::InvalidSyntaxException( const String& filename, unsigned long lineNumber, long column ) : AxisParserException(filename, lineNumber, column)
{
	// no more ops here
}

InvalidSyntaxException::InvalidSyntaxException( const String& message, String& filename, unsigned long lineNumber, long column ) : AxisParserException(message, filename, lineNumber, column)
{
	// no more ops here
}

InvalidSyntaxException::InvalidSyntaxException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisParserException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

InvalidSyntaxException::InvalidSyntaxException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisParserException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::InvalidSyntaxException::InvalidSyntaxException( const InvalidSyntaxException& ex )
{
	Copy(ex);
}

InvalidSyntaxException& axis::foundation::InvalidSyntaxException::operator=( const InvalidSyntaxException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InvalidSyntaxException::Clone( void ) const
{
	return *new InvalidSyntaxException(*this);
}

axis::String axis::foundation::InvalidSyntaxException::GetTypeName( void ) const
{
	return _T("InvalidSyntaxException");
}