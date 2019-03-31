#include "DefaultMessageFilter.hpp"


axis::services::messaging::filters::DefaultMessageFilter::DefaultMessageFilter( void )
{
	// nothing to do here
}

axis::services::messaging::filters::DefaultMessageFilter::~DefaultMessageFilter( void )
{
	// nothing to do here
}

bool axis::services::messaging::filters::DefaultMessageFilter::IsEventMessageFiltered( const axis::services::messaging::EventMessage& message )
{
	return false;
}

bool axis::services::messaging::filters::DefaultMessageFilter::IsResultMessageFiltered( const axis::services::messaging::ResultMessage& message )
{
	return false;
}

void axis::services::messaging::filters::DefaultMessageFilter::Destroy( void ) const
{
	delete this;
}

axis::services::messaging::filters::MessageFilter& axis::services::messaging::filters::DefaultMessageFilter::Clone( void ) const
{
	return *new DefaultMessageFilter();
}
