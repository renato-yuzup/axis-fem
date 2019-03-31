#include "IteratorLogic.hpp"

namespace asli = axis::services::language::iterators;

asli::IteratorLogic::~IteratorLogic( void )
{
	/* Default implementation -- nothing to do */
}

asli::IteratorLogic::IteratorLogic( void )
{
	_useCount = 0;
}

void asli::IteratorLogic::NotifyUse( void )
{
	++_useCount;
}

void asli::IteratorLogic::NotifyDestroy( void )
{
	--_useCount;
	if (_useCount <= 0)
	{
		delete this;
	}
}