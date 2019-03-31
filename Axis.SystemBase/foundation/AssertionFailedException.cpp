#include "AssertionFailedException.hpp"

using namespace axis::foundation;
using namespace axis;

AssertionFailedException::AssertionFailedException(void) : ApplicationErrorException()
{
	// no more ops here
}

AssertionFailedException::AssertionFailedException( const String& message, const AxisException * innerException ) : ApplicationErrorException(message, innerException)
{
	// no more ops here
}

AssertionFailedException::AssertionFailedException( const AxisException * innerException ) : ApplicationErrorException(innerException)
{
	// no more ops here
}

AssertionFailedException::AssertionFailedException( const String& message ) : ApplicationErrorException(message)
{
	// no more ops here
}

axis::foundation::AssertionFailedException::AssertionFailedException( const AssertionFailedException& ex )
{
	Copy(ex);
}

axis::foundation::AssertionFailedException::AssertionFailedException( const AsciiString& expression, const AsciiString& functionName, const AsciiString& filename, long lineNumber )
{
	String msg, buf;

	StringEncoding::AssignFromASCII(expression.data(), buf);
	msg = _T("Assertion for expression:\n\n") + buf + _T("\n\nfailed ");

	StringEncoding::AssignFromASCII(functionName.data(), buf);
	msg += _T("at function ") + buf + _T(" on source file: \n");
	StringEncoding::AssignFromASCII(filename.data(), buf);
	msg += buf + _T(", line ") + String::int_parse(lineNumber);
	SetMessage(msg);
}

AssertionFailedException& axis::foundation::AssertionFailedException::operator=( const AssertionFailedException& ex )
{
	Copy(ex);
	return *this;
}

AxisException& axis::foundation::AssertionFailedException::Clone( void ) const
{
	return *new AssertionFailedException(*this);
}

axis::String axis::foundation::AssertionFailedException::GetTypeName( void ) const
{
	return _T("AssertionFailedException");
}