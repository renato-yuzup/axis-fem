#include "Level2ParserException.hpp"

using namespace axis::foundation;
using namespace axis;

Level2ParserException::Level2ParserException(void) : AxisParserException()
{
	// no more ops here
}

Level2ParserException::Level2ParserException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	// no more ops here
}

Level2ParserException::Level2ParserException( const AxisException * innerException ) : AxisParserException(innerException)
{
	// no more ops here
}

Level2ParserException::Level2ParserException( const String& message ) : AxisParserException(message)
{
	// no more ops here
}

axis::foundation::Level2ParserException::Level2ParserException( const Level2ParserException& ex )
{
	Copy(ex);
}

axis::foundation::Level2ParserException axis::foundation::Level2ParserException::operator=( const Level2ParserException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::Level2ParserException::Clone( void ) const
{
	return *new Level2ParserException(*this);
}

axis::String axis::foundation::Level2ParserException::GetTypeName( void ) const
{
	return _T("Level2ParserException");
}