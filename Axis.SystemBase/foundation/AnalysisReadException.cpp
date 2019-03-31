#include "AnalysisReadException.hpp"

using namespace axis::foundation;
using namespace axis;

AnalysisReadException::AnalysisReadException(void) : AxisException()
{
	// no more ops here
}

AnalysisReadException::AnalysisReadException( const String& message, const AxisException *innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

AnalysisReadException::AnalysisReadException( const AxisException *innerException ) : AxisException(innerException)
{
	// no more ops here
}

AnalysisReadException::AnalysisReadException( const String& message ) : AxisException(message)
{
	// no more ops here
}


AxisException& axis::foundation::AnalysisReadException::Clone( void ) const
{
	return *new AnalysisReadException(*this);
}

axis::foundation::AnalysisReadException::AnalysisReadException( const AnalysisReadException& exception ) : AxisException(exception)
{
	// no more ops here
}

AnalysisReadException& axis::foundation::AnalysisReadException::operator=( const AnalysisReadException& e )
{
	Copy(e);
	return *this;
}

axis::String axis::foundation::AnalysisReadException::GetTypeName( void ) const
{
	return _T("AnalysisReadException");
}