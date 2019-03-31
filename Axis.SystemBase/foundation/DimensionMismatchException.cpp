#include "DimensionMismatchException.hpp"

using namespace axis::foundation;
using namespace axis;

DimensionMismatchException::DimensionMismatchException(void) : AxisException()
{
	// no more ops here
}

DimensionMismatchException::DimensionMismatchException(String &location) : AxisException()
{
  // no more ops here
}

DimensionMismatchException::DimensionMismatchException(const char_type *location) : AxisException()
{
  // no more ops here
}

DimensionMismatchException::DimensionMismatchException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

DimensionMismatchException::DimensionMismatchException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

axis::foundation::DimensionMismatchException::DimensionMismatchException( const DimensionMismatchException& ex )
{
	Copy(ex);
}

DimensionMismatchException& axis::foundation::DimensionMismatchException::operator=( const DimensionMismatchException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::DimensionMismatchException::Clone( void ) const
{
	return *new DimensionMismatchException(*this);
}

axis::String axis::foundation::DimensionMismatchException::GetTypeName( void ) const
{
	return _T("DimensionMismatchException");
}