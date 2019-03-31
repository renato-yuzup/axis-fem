#include "MessageListener.hpp"

axis::services::messaging::MessageListener::MessageListener( void )
{
	_filter = &axis::services::messaging::filters::MessageFilter::Default.Clone();
}

axis::services::messaging::MessageListener::MessageListener( const axis::services::messaging::filters::MessageFilter& filter )
{
	_filter = &filter.Clone();
}

axis::services::messaging::MessageListener::~MessageListener( void )
{
	_filter->Destroy();
}

void axis::services::messaging::MessageListener::ProcessMessage( Message& volatileMessage )
{
	if (volatileMessage.IsEvent())
	{
		if (!_filter->IsEventMessageFiltered((EventMessage&)volatileMessage))
		{
			DoProcessEventMessage((EventMessage&)volatileMessage);
		}
	}
	else
	{
		if (!_filter->IsResultMessageFiltered((ResultMessage&)volatileMessage))
		{
			DoProcessResultMessage((ResultMessage&)volatileMessage);
		}
	}
}

void axis::services::messaging::MessageListener::DoProcessEventMessage( EventMessage& volatileMessage )
{
	// base implementation does nothing; it is up to child classes to do something
}

void axis::services::messaging::MessageListener::DoProcessResultMessage( ResultMessage& volatileMessage )
{
	// base implementation does nothing; it is up to child classes to do something
}

axis::services::messaging::filters::MessageFilter& axis::services::messaging::MessageListener::GetFilter( void ) const
{
	return *_filter;
}

void axis::services::messaging::MessageListener::ReplaceFilter( const axis::services::messaging::filters::MessageFilter& filter )
{
	if (_filter == &filter) return;
	axis::services::messaging::filters::MessageFilter& f = filter.Clone();
	_filter->Destroy();
	_filter = &f;
}