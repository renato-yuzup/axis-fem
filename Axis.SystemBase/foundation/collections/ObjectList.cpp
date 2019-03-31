#include "ObjectList.hpp"
#include "ObjectListImpl.hpp"


axis::foundation::collections::ObjectList::~ObjectList( void )
{
	// nothing to do here
}

axis::foundation::collections::ObjectList& axis::foundation::collections::ObjectList::Create( void )
{
	return *new ObjectListImpl();
}