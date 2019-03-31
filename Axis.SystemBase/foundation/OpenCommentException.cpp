#include "OpenCommentException.hpp"

using namespace axis::foundation;
using namespace axis;

OpenCommentException::OpenCommentException(void) : AxisParserException()
{
	// no more ops here
}

OpenCommentException::OpenCommentException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

OpenCommentException::OpenCommentException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

OpenCommentException::OpenCommentException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}


axis::foundation::OpenCommentException::OpenCommentException( const String& filename, unsigned long lineNumber ) : AxisParserException(filename, lineNumber)
{
	// no more ops here
}

axis::foundation::OpenCommentException::OpenCommentException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisParserException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::OpenCommentException::OpenCommentException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisParserException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::OpenCommentException::OpenCommentException( const OpenCommentException& ex )
{
	Copy(ex);
}

OpenCommentException& axis::foundation::OpenCommentException::operator=( const OpenCommentException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::OpenCommentException::Clone( void ) const
{
	return *new OpenCommentException(*this);
}

axis::String axis::foundation::OpenCommentException::GetTypeName( void ) const
{
	return _T("OpenCommentException");
}

OpenCommentException::OpenCommentException( const String& filename, unsigned long lineNumber, long column ) : AxisParserException(filename, lineNumber, column)
{
	// no more ops here
}

OpenCommentException::OpenCommentException( const String& message, String& filename, unsigned long lineNumber, long column ) : AxisParserException(message, filename, lineNumber, column)
{
	// no more ops here
}
