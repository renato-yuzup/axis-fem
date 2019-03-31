#include "ResultCollectionBlockSyntax.hpp"

namespace afd = axis::foundation::definitions;

const axis::String::char_type * afd::ResultCollectionBlockSyntax::BlockName = _T("OUTPUT");
const axis::String::char_type * afd::ResultCollectionBlockSyntax::FileFormatParameterName = _T("FORMAT");
const axis::String::char_type * afd::ResultCollectionBlockSyntax::AppendParameterName = _T("APPEND");
const axis::String::char_type * afd::ResultCollectionBlockSyntax::FileNameParameterName = _T("FILE");
const axis::String::char_type * afd::ResultCollectionBlockSyntax::FormatArgumentsParameterName = _T("FORMAT_PARAMS");

afd::ResultCollectionBlockSyntax::ResultCollectionBlockSyntax( void )
{
	// nothing to do here
}

