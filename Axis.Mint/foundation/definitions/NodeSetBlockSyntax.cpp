#include "NodeSetBlockSyntax.hpp"

namespace afd = axis::foundation::definitions;

afd::NodeSetBlockSyntax::NodeSetBlockSyntax( void )
{
	// nothing to do here
}

afd::NodeSetBlockSyntax::~NodeSetBlockSyntax( void )
{
	// nothing to do here
}

const axis::String::char_type * afd::NodeSetBlockSyntax::SetIdAttributeName = _T("ID");
const axis::String::char_type * afd::NodeSetBlockSyntax::BlockName = _T("NODE_SET");
