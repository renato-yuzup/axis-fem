#include "ArenaException.hpp"


using namespace axis::foundation;
using namespace axis;

ArenaException::ArenaException(void) : ApplicationErrorException()
{
	// no more ops here
}

ArenaException::ArenaException( const String& message, const AxisException * innerException ) : ApplicationErrorException(message, innerException)
{
	// no more ops here
}

ArenaException::ArenaException( const AxisException * innerException ) : ApplicationErrorException(innerException)
{
	// no more ops here
}

ArenaException::ArenaException( const String& message ) : ApplicationErrorException(message)
{
	// no more ops here
}

axis::foundation::ArenaException::ArenaException( const ArenaException& ex )
{
	Copy(ex);
}

ArenaException& axis::foundation::ArenaException::operator=( const ArenaException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::ArenaException::Clone( void ) const
{
	return *new ArenaException(*this);
}

axis::String axis::foundation::ArenaException::GetTypeName( void ) const
{
	return _T("ArenaException");
}