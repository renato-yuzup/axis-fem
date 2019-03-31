#include "InvalidPreProcessorDirectiveException.hpp"

using namespace axis::foundation;
using namespace axis;

InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException(void) : PreProcessorException()
{
	// no more ops here
}

InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const String& message, const AxisException * innerException ) : PreProcessorException(message, innerException)
{
	// no more ops here
}

InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const AxisException * innerException ) : PreProcessorException(innerException)
{
	// no more ops here
}

InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const String& message ) : PreProcessorException(message)
{
	// no more ops here
}


InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const String& filename, unsigned long lineNumber ) : PreProcessorException(filename, lineNumber)
{
	// no more ops here
}
InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(filename, lineNumber, column)
{
	// no more ops here
}

InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const String& message, String& filename, unsigned long lineNumber, long column ) : PreProcessorException(message, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

axis::foundation::InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : PreProcessorException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::InvalidPreProcessorDirectiveException::InvalidPreProcessorDirectiveException( const InvalidPreProcessorDirectiveException& ex )
{
	Copy(ex);
}

AxisException& axis::foundation::InvalidPreProcessorDirectiveException::Clone( void ) const
{
	return *new InvalidPreProcessorDirectiveException(*this);
}

InvalidPreProcessorDirectiveException& axis::foundation::InvalidPreProcessorDirectiveException::operator=( const InvalidPreProcessorDirectiveException& ex )
{
	Copy(ex);
	return *this;
}

axis::String axis::foundation::InvalidPreProcessorDirectiveException::GetTypeName( void ) const
{
	return _T("InvalidPreProcessorDirectiveException");
}