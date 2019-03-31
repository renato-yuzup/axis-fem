#include "ArenaStackMismatchException.hpp"

using axis::foundation::ArenaStackMismatchException;
namespace af = axis::foundation;
using namespace axis;

ArenaStackMismatchException::ArenaStackMismatchException(void) : ArenaException()
{
  // no more ops here
}

ArenaStackMismatchException::ArenaStackMismatchException( 
  const String& message, 
  const AxisException * innerException ) : ArenaException(message, innerException)
{
  // no more ops here
}

ArenaStackMismatchException::ArenaStackMismatchException( 
  const AxisException * innerException ) : ArenaException(innerException)
{
  // no more ops here
}

ArenaStackMismatchException::ArenaStackMismatchException( const String& message ) : 
  ArenaException(message)
{
  // no more ops here
}

ArenaStackMismatchException::ArenaStackMismatchException( 
  const ArenaStackMismatchException& ex )
{
  Copy(ex);
}

ArenaStackMismatchException& ArenaStackMismatchException::operator =( 
  const ArenaStackMismatchException& ex )
{
  Copy(ex);
  return *this;
}

af::AxisException& ArenaStackMismatchException::Clone( void ) const
{
  return *new ArenaStackMismatchException(*this);
}

axis::String ArenaStackMismatchException::GetTypeName( void ) const
{
  return _T("ArenaStackMismatchException");
}

