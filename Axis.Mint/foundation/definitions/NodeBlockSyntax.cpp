#include "NodeBlockSyntax.hpp"

namespace afd = axis::foundation::definitions;

afd::NodeBlockSyntax::NodeBlockSyntax( void )
{
	// nothing to do here
}

afd::NodeBlockSyntax::~NodeBlockSyntax( void )
{
	// nothing to do here
}

const axis::String::char_type * afd::NodeBlockSyntax::SetIdAttributeName = _T("SET_ID");
const axis::String::char_type * afd::NodeBlockSyntax::BlockName = _T("NODES");
