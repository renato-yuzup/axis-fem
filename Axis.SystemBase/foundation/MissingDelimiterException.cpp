#include "MissingDelimiterException.hpp"

using namespace axis::foundation;
using namespace axis;

MissingDelimiterException::MissingDelimiterException(void) : AxisParserException()
{
	// no more ops here
}

MissingDelimiterException::MissingDelimiterException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

MissingDelimiterException::MissingDelimiterException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

MissingDelimiterException::MissingDelimiterException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}


MissingDelimiterException::MissingDelimiterException( const String& filename, unsigned long lineNumber ) : AxisParserException(filename, lineNumber)
{
	// no more ops here
}
MissingDelimiterException::MissingDelimiterException( const String& filename, unsigned long lineNumber, long column ) : AxisParserException(filename, lineNumber, column)
{
	// no more ops here
}

MissingDelimiterException::MissingDelimiterException( const String& message, String& filename, unsigned long lineNumber, long column ) : AxisParserException(message, filename, lineNumber, column)
{
	// no more ops here
}

MissingDelimiterException::MissingDelimiterException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisParserException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

MissingDelimiterException::MissingDelimiterException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisParserException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::MissingDelimiterException::MissingDelimiterException( const MissingDelimiterException& ex )
{
	Copy(ex);
}

axis::foundation::MissingDelimiterException axis::foundation::MissingDelimiterException::operator=( const MissingDelimiterException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::MissingDelimiterException::Clone( void ) const
{
	return *new MissingDelimiterException(*this);
}

axis::String axis::foundation::MissingDelimiterException::GetTypeName( void ) const
{
	return _T("MissingDelimiterException");
}