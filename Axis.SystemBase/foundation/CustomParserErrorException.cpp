#include "CustomParserErrorException.hpp"

using namespace axis::foundation;
using namespace axis;

CustomParserErrorException::CustomParserErrorException(void) : AxisParserException()
{
	_errorCode = 0;
}

CustomParserErrorException::CustomParserErrorException( const String& message, const AxisException * innerException ) : AxisParserException(message, innerException)
{
	_errorCode = 0;
}

CustomParserErrorException::CustomParserErrorException( const AxisException * innerException ) : AxisParserException(innerException)
{
	_errorCode = 0;
}

CustomParserErrorException::CustomParserErrorException( const String& message ) : AxisParserException(message)
{
	_errorCode = 0;
}

axis::String axis::foundation::CustomParserErrorException::GetBlockName( void ) const
{
	return (_blockName);
}

int axis::foundation::CustomParserErrorException::GetCustomErrorCode( void ) const
{
	return _errorCode;
}

axis::foundation::CustomParserErrorException& axis::foundation::CustomParserErrorException::SetBlockName( const axis::String& blockName )
{
	_blockName = blockName;
	return *this;
}

axis::foundation::CustomParserErrorException& axis::foundation::CustomParserErrorException::SetCustomErrorCode( int errorCode )
{
	_errorCode = errorCode;
	return *this;
}

axis::foundation::CustomParserErrorException::CustomParserErrorException( const CustomParserErrorException& ex )
{
	Copy(ex);
	_blockName = ex._blockName;
	_errorCode = ex._errorCode;
}

CustomParserErrorException& axis::foundation::CustomParserErrorException::operator=( const CustomParserErrorException& ex )
{
	Copy(ex);
	_blockName = ex._blockName;
	_errorCode = ex._errorCode;
	return *this;
}

AxisException& axis::foundation::CustomParserErrorException::Clone( void ) const
{
	return *new CustomParserErrorException(*this);
}

axis::String axis::foundation::CustomParserErrorException::GetTypeName( void ) const
{
	return _T("CustomParserErrorException");
}