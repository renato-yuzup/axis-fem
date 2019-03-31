#include "PreProcessorException.hpp"

using namespace axis::foundation;
using namespace axis;

PreProcessorException::PreProcessorException(void) : AxisParserException()
{
	// no more ops here
}

PreProcessorException::PreProcessorException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

PreProcessorException::PreProcessorException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

PreProcessorException::PreProcessorException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}


PreProcessorException::PreProcessorException( const String& filename, unsigned long lineNumber ) : AxisParserException(filename, lineNumber)
{
	// no more ops here
}
PreProcessorException::PreProcessorException( const String& filename, unsigned long lineNumber, long column ) : AxisParserException(filename, lineNumber, column)
{
	// no more ops here
}

PreProcessorException::PreProcessorException( const String& message, String& filename, unsigned long lineNumber, long column ) : AxisParserException(message, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::PreProcessorException::PreProcessorException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisParserException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::PreProcessorException::PreProcessorException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisParserException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::PreProcessorException::PreProcessorException( const PreProcessorException& ex )
{
	Copy(ex);
}

PreProcessorException& axis::foundation::PreProcessorException::operator=( const PreProcessorException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::PreProcessorException::Clone( void ) const
{
	return *new PreProcessorException(*this);
}

axis::String axis::foundation::PreProcessorException::GetTypeName( void ) const
{
	return _T("PreProcessorException");
}