#include "UnexpectedExpressionException.hpp"

using namespace axis::foundation;
using namespace axis;

UnexpectedExpressionException::UnexpectedExpressionException(void) : AxisParserException()
{
	// no more ops here
}

UnexpectedExpressionException::UnexpectedExpressionException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

UnexpectedExpressionException::UnexpectedExpressionException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

UnexpectedExpressionException::UnexpectedExpressionException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}


UnexpectedExpressionException::UnexpectedExpressionException( const String& filename, unsigned long lineNumber ) : AxisParserException(filename, lineNumber)
{
	// no more ops here
}
UnexpectedExpressionException::UnexpectedExpressionException( const String& filename, unsigned long lineNumber, long column ) : AxisParserException(filename, lineNumber, column)
{
	// no more ops here
}

UnexpectedExpressionException::UnexpectedExpressionException( const String& message, String& filename, unsigned long lineNumber, long column ) : AxisParserException(message, filename, lineNumber, column)
{
	// no more ops here
}

UnexpectedExpressionException::UnexpectedExpressionException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisParserException(innerException, filename, lineNumber)
{
	// no more ops here
}

UnexpectedExpressionException::UnexpectedExpressionException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisParserException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::UnexpectedExpressionException::UnexpectedExpressionException( const UnexpectedExpressionException& ex )
{
	Copy(ex);
}

UnexpectedExpressionException& axis::foundation::UnexpectedExpressionException::operator=( const UnexpectedExpressionException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::UnexpectedExpressionException::Clone( void ) const
{
	return *new UnexpectedExpressionException(*this);
}

axis::String axis::foundation::UnexpectedExpressionException::GetTypeName( void ) const
{
	return _T("UnexpectedExpressionException");
}