#include "ElementSetBlockSyntax.hpp"

namespace afd = axis::foundation::definitions;

afd::ElementSetBlockSyntax::ElementSetBlockSyntax( void )
{
	// nothing to do here
}

afd::ElementSetBlockSyntax::~ElementSetBlockSyntax( void )
{
	// nothing to do here
}

const axis::String::char_type * afd::ElementSetBlockSyntax::SetIdAttributeName = _T("ID");
const axis::String::char_type * afd::ElementSetBlockSyntax::BlockName = _T("ELEMENT_SET");
