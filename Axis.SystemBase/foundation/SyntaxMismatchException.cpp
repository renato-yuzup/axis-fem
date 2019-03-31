#include "SyntaxMismatchException.hpp"

using namespace axis::foundation;
using namespace axis;

SyntaxMismatchException::SyntaxMismatchException(void) : AxisException()
{
	// no more ops here
}

SyntaxMismatchException::SyntaxMismatchException( const String& message, const AxisException * innerException ) : AxisException(message, innerException)
{
	// no more ops here
}

SyntaxMismatchException::SyntaxMismatchException( const AxisException * innerException ) : AxisException(innerException)
{
	// no more ops here
}

SyntaxMismatchException::SyntaxMismatchException( const String& message ) : AxisException(message)
{
	// no more ops here
}

int axis::foundation::SyntaxMismatchException::GetLine( void )
{
	return _lineNumber;
}

void axis::foundation::SyntaxMismatchException::SetLine( int value )
{
	_lineNumber = value;
}

const String axis::foundation::SyntaxMismatchException::GetFileName( void )
{
	return _fileName;
}

void axis::foundation::SyntaxMismatchException::SetFileName( const String filename )
{
	_fileName = filename;
}

void axis::foundation::SyntaxMismatchException::SetFileName( const char_type *filename )
{
	_fileName = filename;
}

const String axis::foundation::SyntaxMismatchException::GetMessageReason( void )
{
	return _reason;
}

void axis::foundation::SyntaxMismatchException::SetMessageReason( const String message )
{
	_reason = message;
}

void axis::foundation::SyntaxMismatchException::SetMessageReason( const char_type *message )
{
	_reason = message;
}

void axis::foundation::SyntaxMismatchException::Copy( const AxisException& e )
{
	SyntaxMismatchException& ex = (SyntaxMismatchException&)e;
	AxisException::Copy(e);
	_fileName = ex._fileName;
	_reason = ex._reason;
	_lineNumber = ex._lineNumber;
}

axis::foundation::SyntaxMismatchException::SyntaxMismatchException( const SyntaxMismatchException& ex )
{
	Copy(ex);
}

SyntaxMismatchException& axis::foundation::SyntaxMismatchException::operator=( const SyntaxMismatchException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::SyntaxMismatchException::Clone( void ) const
{
	return *new SyntaxMismatchException(*this);
}

axis::String axis::foundation::SyntaxMismatchException::GetTypeName( void ) const
{
	return _T("SyntaxMismatchException");
}