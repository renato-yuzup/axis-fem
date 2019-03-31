#include "AxisParserException.hpp"

using namespace axis::foundation;
using namespace axis;

AxisParserException::AxisParserException(void) : AxisException()
{
	// no more ops here
}

AxisParserException::AxisParserException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

AxisParserException::AxisParserException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

AxisParserException::AxisParserException( const String& message ) : AxisException(message)
{
	// no more ops here
}

AxisParserException::AxisParserException( const String& filename, unsigned long lineNumber, long column ) : AxisException()
{
	_filename = filename;
	_lineNumber = lineNumber;
	_column = column;
}

axis::foundation::AxisParserException::AxisParserException( const String& filename, unsigned long lineNumber ) : AxisException()
{
	_filename = filename;
	_lineNumber = lineNumber;	
}

axis::foundation::AxisParserException::AxisParserException( const AxisException *innerException, const String& filename, unsigned long lineNumber, long column ) : AxisException(innerException)
{
	_filename = filename;
	_lineNumber = lineNumber;
	_column = column;
}

axis::foundation::AxisParserException::AxisParserException( const AxisException *innerException, const String& filename, unsigned long lineNumber ) : AxisException(innerException)
{
	_filename = filename;
	_lineNumber = lineNumber;
}

axis::foundation::AxisParserException::AxisParserException( const AxisParserException& ex )
{
	Copy(ex);
	_filename = ex._filename;
	_lineNumber = ex._lineNumber;
	_column = ex._column;
}

AxisParserException::AxisParserException( const String& message, const String& filename, unsigned long lineNumber, long column ) : AxisException(message)
{
	_filename = filename;
	_lineNumber = lineNumber;
	_column = column;
}

bool axis::foundation::AxisParserException::IsColumnDefined( void ) const
{
	return (_column > 0);
}

long axis::foundation::AxisParserException::GetColumnIndex( void ) const
{
	return _column;
}

unsigned long axis::foundation::AxisParserException::GetLineIndex( void ) const
{
	return _lineNumber;
}

String axis::foundation::AxisParserException::GetFileName( void ) const
{
	return _filename;
}

AxisParserException& axis::foundation::AxisParserException::operator=( const AxisParserException& ex )
{
	Copy(ex);
	_filename = ex._filename;
	_lineNumber = ex._lineNumber;
	_column = ex._column;
	return *this;
}

AxisException& axis::foundation::AxisParserException::Clone( void ) const
{
	return *new AxisParserException(*this);
}

void axis::foundation::AxisParserException::Copy( const AxisException& ex )
{
	AxisParserException& e = (AxisParserException&)ex;
	AxisException::Copy(ex);
	_lineNumber = e._lineNumber;
	_column = e._column;
	_filename = e._filename;
}

axis::String axis::foundation::AxisParserException::GetTypeName( void ) const
{
	return _T("AxisParserException");
}