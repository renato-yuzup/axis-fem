#include "InvalidIdentifierException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidIdentifierException::InvalidIdentifierException(void) : AxisParserException()
{
	// no more ops here
}

InvalidIdentifierException::InvalidIdentifierException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

InvalidIdentifierException::InvalidIdentifierException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

InvalidIdentifierException::InvalidIdentifierException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}


InvalidIdentifierException::InvalidIdentifierException( const String& filename, unsigned long lineNumber ) : AxisParserException(filename, lineNumber)
{
	// no more ops here
}
InvalidIdentifierException::InvalidIdentifierException( const String& filename, unsigned long lineNumber, long column ) : AxisParserException(filename, lineNumber, column)
{
	// no more ops here
}

InvalidIdentifierException::InvalidIdentifierException( const String& message, String& filename, unsigned long lineNumber, long column ) : AxisParserException(message, filename, lineNumber, column)
{
	// no more ops here
}

InvalidIdentifierException::InvalidIdentifierException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisParserException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

InvalidIdentifierException::InvalidIdentifierException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisParserException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::InvalidIdentifierException::InvalidIdentifierException( const InvalidIdentifierException& ex )
{
	Copy(ex);
}

InvalidIdentifierException& axis::foundation::InvalidIdentifierException::operator=( const InvalidIdentifierException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InvalidIdentifierException::Clone( void ) const
{
	return *new InvalidIdentifierException(*this);
}

axis::String axis::foundation::InvalidIdentifierException::GetTypeName( void ) const
{
	return _T("InvalidIdentifierException");
}