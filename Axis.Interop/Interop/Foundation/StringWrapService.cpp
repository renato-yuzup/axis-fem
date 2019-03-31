#include "StdAfx.h"
#include "StringWrapService.hpp"

typedef axis::String::value_type char_type;

axis::String axis::Interop::foundation::StringWrapService::WrapToAxisString( System::String ^str )
{

	char_type* chars = new char_type[str->Length + 1];
	for (int i = 0; i < str->Length; i++)
	{
		chars[i] = (char_type)((*str)[i]);
	}
	chars[str->Length] = NULL;
	axis::String retVal(chars);
	delete [] chars;
	return retVal;
}

System::String ^axis::Interop::foundation::StringWrapService::UnwrapFromAxisString( const axis::String& str )
{
	return gcnew System::String(str.data());
}

axis::Interop::foundation::StringWrapService::StringWrapService( void )
{
	// nothing to do
}