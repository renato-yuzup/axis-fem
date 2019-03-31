#include "SymbolRedefinedException.hpp"

using namespace axis::foundation;
using namespace axis;

SymbolRedefinedException::SymbolRedefinedException(void) : PreProcessorException()
{
	// no more ops here
}

SymbolRedefinedException::SymbolRedefinedException( const String& message, const AxisException * innerException ) : PreProcessorException(message, innerException)
{
	// no more ops here
}

SymbolRedefinedException::SymbolRedefinedException( const AxisException * innerException ) : PreProcessorException(innerException)
{
	// no more ops here
}

SymbolRedefinedException::SymbolRedefinedException( const String& message ) : PreProcessorException(message)
{
	// no more ops here
}


SymbolRedefinedException::SymbolRedefinedException( const String& filename, unsigned long lineNumber ) : PreProcessorException(filename, lineNumber)
{
	// no more ops here
}
SymbolRedefinedException::SymbolRedefinedException( const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(filename, lineNumber, column)
{
	// no more ops here
}

SymbolRedefinedException::SymbolRedefinedException( const String& message, String& filename, unsigned long lineNumber, long column ) : PreProcessorException(message, filename, lineNumber, column)
{
	// no more ops here
}

SymbolRedefinedException::SymbolRedefinedException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : PreProcessorException(innerException, filename, lineNumber, column)
{
	// no more ops here
}

SymbolRedefinedException::SymbolRedefinedException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : PreProcessorException(innerException, filename, lineNumber)
{
	// no more ops here
}

axis::foundation::SymbolRedefinedException::SymbolRedefinedException( const SymbolRedefinedException& ex )
{
	Copy(ex);
}

SymbolRedefinedException& axis::foundation::SymbolRedefinedException::operator=( const SymbolRedefinedException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::SymbolRedefinedException::Clone( void ) const
{
	return *new SymbolRedefinedException(*this);
}

axis::String axis::foundation::SymbolRedefinedException::GetTypeName( void ) const
{
	return _T("SymbolRedefinedException");
}