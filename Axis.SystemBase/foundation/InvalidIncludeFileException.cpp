#include "InvalidIncludeFileException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidIncludeFileException::InvalidIncludeFileException(void) : PreProcessorException()
{
	// no more ops here
}

InvalidIncludeFileException::InvalidIncludeFileException( const String& message, const AxisException * innerException ) : PreProcessorException(message, innerException)
{
	// no more ops here
}

InvalidIncludeFileException::InvalidIncludeFileException( const AxisException * innerException ) : PreProcessorException(innerException)
{
	// no more ops here
}

InvalidIncludeFileException::InvalidIncludeFileException( const String& message ) : PreProcessorException(message)
{
	// no more ops here
}


InvalidIncludeFileException::InvalidIncludeFileException( const String& filename, unsigned long lineNumber ) : PreProcessorException(filename, lineNumber)
{
	// no more ops here
}
InvalidIncludeFileException::InvalidIncludeFileException( const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(filename, lineNumber, column)
{
	// no more ops here
}

InvalidIncludeFileException::InvalidIncludeFileException( const String& message, String& filename, unsigned long lineNumber, long column ) : PreProcessorException(message, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::InvalidIncludeFileException::InvalidIncludeFileException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : PreProcessorException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::InvalidIncludeFileException::InvalidIncludeFileException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::InvalidIncludeFileException::InvalidIncludeFileException( const InvalidIncludeFileException& ex )
{
	Copy(ex);
}

InvalidIncludeFileException& axis::foundation::InvalidIncludeFileException::operator=( const InvalidIncludeFileException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InvalidIncludeFileException::Clone( void ) const
{
	return *new InvalidIncludeFileException(*this);
}

axis::String axis::foundation::InvalidIncludeFileException::GetTypeName( void ) const
{
	return _T("InvalidIncludeFileException");
}