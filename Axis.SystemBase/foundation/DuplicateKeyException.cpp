#include "DuplicateKeyException.hpp"

using namespace axis::foundation;
using namespace axis;

DuplicateKeyException::DuplicateKeyException(void) : AxisException()
{
  // no more ops here
}

DuplicateKeyException::DuplicateKeyException(String &location) : AxisException()
{
  // no more ops here
}

DuplicateKeyException::DuplicateKeyException(const char_type *location) : AxisException()
{
  // no more ops here
}

DuplicateKeyException::DuplicateKeyException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
  // no more ops here
}

DuplicateKeyException::DuplicateKeyException( const AxisException * innerException ) : AxisException(innerException)
{
  // no more ops here
}

axis::foundation::DuplicateKeyException::DuplicateKeyException( const DuplicateKeyException& ex )
{
  Copy(ex);
}

DuplicateKeyException& axis::foundation::DuplicateKeyException::operator=( const DuplicateKeyException& ex )
{
  Copy(ex);
  return *this;
}

AxisException& axis::foundation::DuplicateKeyException::Clone( void ) const
{
  return *new DuplicateKeyException(*this);
}

axis::String axis::foundation::DuplicateKeyException::GetTypeName( void ) const
{
  return _T("DuplicateKeyException");
}
