#include "ElementBlockSyntax.hpp"

namespace afd = axis::foundation::definitions;

afd::ElementBlockSyntax::ElementBlockSyntax( void )
{
	// nothing to do here
}

afd::ElementBlockSyntax::~ElementBlockSyntax( void )
{
	// nothing to do here
}

const axis::String::char_type * afd::ElementBlockSyntax::SetIdAttributeName = _T("SET_ID");
const axis::String::char_type * afd::ElementBlockSyntax::BlockName = _T("ELEMENTS");
