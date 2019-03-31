#include "InputStackFullException.hpp"

using namespace axis::foundation;
using namespace axis;

InputStackFullException::InputStackFullException(void) : PreProcessorException()
{
	// no more ops here
}

InputStackFullException::InputStackFullException( const String& message, const AxisException * innerException ) : PreProcessorException(message, innerException)
{
	// no more ops here
}

InputStackFullException::InputStackFullException( const AxisException * innerException ) : PreProcessorException(innerException)
{
	// no more ops here
}

InputStackFullException::InputStackFullException( const String& message ) : PreProcessorException(message)
{
	// no more ops here
}


InputStackFullException::InputStackFullException( const String& filename, unsigned long lineNumber ) : PreProcessorException(filename, lineNumber)
{
	// no more ops here
}
InputStackFullException::InputStackFullException( const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(filename, lineNumber, column)
{
	// no more ops here
}

InputStackFullException::InputStackFullException( const String& message, String& filename, unsigned long lineNumber, long column ) : PreProcessorException(message, filename, lineNumber, column)
{
	// no more ops here
}

InputStackFullException::InputStackFullException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

InputStackFullException::InputStackFullException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : PreProcessorException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::InputStackFullException::InputStackFullException( const InputStackFullException& ex ) : PreProcessorException(ex)
{
	Copy(ex);
}

InputStackFullException& axis::foundation::InputStackFullException::operator=( const InputStackFullException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::InputStackFullException::Clone( void ) const
{
	return *new InputStackFullException(*this);
}

axis::String axis::foundation::InputStackFullException::GetTypeName( void ) const
{
	return _T("InputStackFullException");
}