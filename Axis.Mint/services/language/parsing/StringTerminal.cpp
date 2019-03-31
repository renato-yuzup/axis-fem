#include "StringTerminal.hpp"

namespace aslp = axis::services::language::parsing;

aslp::StringTerminal::StringTerminal( const axis::String& value ) :
_value(value)
{
	// nothing more to do
}

bool aslp::StringTerminal::IsId( void ) const
{
	return false;
}

bool aslp::StringTerminal::IsNumber( void ) const
{
	return false;
}

bool aslp::StringTerminal::IsString( void ) const
{
	return true;
}

bool aslp::StringTerminal::IsReservedWord( void ) const
{
	return false;
}

bool aslp::StringTerminal::IsOperator( void ) const
{
	return false;
}

aslp::ParseTreeNode& aslp::StringTerminal::Clone( void ) const
{
	return *new StringTerminal(_value);
}

axis::String aslp::StringTerminal::ToString( void ) const
{
	return _value;
}

axis::String aslp::StringTerminal::InsertEscapeChars( const axis::String& s ) const
{
	axis::String str = _value;
	axis::String::size_type pos;
	const int escapeCharsLen = 2;
	axis::String escapeChars[escapeCharsLen];
	axis::String escapeCharsStr[escapeCharsLen];
	escapeChars[0] = _T("\t");
	escapeCharsStr[0] = _T("\\t");
	escapeChars[1] = _T("\"");
	escapeCharsStr[1] = _T("\\\"");
	for (int i = 0; i < escapeCharsLen; i++)
	{
		while((pos = str.find(escapeChars[i])) != axis::String::npos)
		{
			str.replace(pos, escapeChars[i].size(), escapeCharsStr[i]);
		}
	}
	return str;
}

axis::String aslp::StringTerminal::ToExpressionString( void ) const
{
	axis::String s = _T("\"");
	s.append(InsertEscapeChars(_value)).append(_T("\""));
	return s;
}