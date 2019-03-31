#include "ArenaMemoryExhaustedException.hpp"

using axis::foundation::ArenaMemoryExhaustedException;
namespace af = axis::foundation;
using namespace axis;

ArenaMemoryExhaustedException::ArenaMemoryExhaustedException(void) : ArenaException()
{
  // no more ops here
}

ArenaMemoryExhaustedException::ArenaMemoryExhaustedException( 
      const String& message, 
      const AxisException * innerException ) : ArenaException(message, innerException)
{
  // no more ops here
}

ArenaMemoryExhaustedException::ArenaMemoryExhaustedException( 
      const AxisException * innerException ) : ArenaException(innerException)
{
  // no more ops here
}

ArenaMemoryExhaustedException::ArenaMemoryExhaustedException( const String& message ) : 
      ArenaException(message)
{
  // no more ops here
}

ArenaMemoryExhaustedException::ArenaMemoryExhaustedException( 
      const ArenaMemoryExhaustedException& ex )
{
  Copy(ex);
}

ArenaMemoryExhaustedException& ArenaMemoryExhaustedException::operator =( 
      const ArenaMemoryExhaustedException& ex )
{
  Copy(ex);
  return *this;
}

af::AxisException& ArenaMemoryExhaustedException::Clone( void ) const
{
  return *new ArenaMemoryExhaustedException(*this);
}

axis::String ArenaMemoryExhaustedException::GetTypeName( void ) const
{
  return _T("ArenaMemoryExhaustedException");
}

