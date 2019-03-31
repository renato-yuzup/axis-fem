#include "CollectorHub.hpp"

axis::services::messaging::CollectorHub::CollectorHub( void )
{
	// nothing to do
}

axis::services::messaging::CollectorHub::~CollectorHub( void )
{
	// nothing to do
}

void axis::services::messaging::CollectorHub::DoProcessEventMessage( EventMessage& volatileMessage )
{
	// stamp and forward this message
	StampEventMessage(volatileMessage);
	ProcessEventMessageLocally(volatileMessage);
	DispatchMessage(volatileMessage);
}

void axis::services::messaging::CollectorHub::DoProcessResultMessage( ResultMessage& volatileMessage )
{
	// stamp and forward this message
	StampResultMessage(volatileMessage);
	ProcessResultMessageLocally(volatileMessage);
	DispatchMessage(volatileMessage);
}

void axis::services::messaging::CollectorHub::StampEventMessage( EventMessage& message ) const
{
	// our default behavior is do nothing
}

void axis::services::messaging::CollectorHub::StampResultMessage( ResultMessage& message ) const
{
	// our default behavior is do nothing
}

void axis::services::messaging::CollectorHub::ProcessEventMessageLocally( const EventMessage& volatileMessage )
{
	// in the default implementation, we do nothing
}

void axis::services::messaging::CollectorHub::ProcessResultMessageLocally( const ResultMessage& volatileMessage )
{
	// in the default implementation, we do nothing
}
