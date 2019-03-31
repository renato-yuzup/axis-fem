#include "CollectorEndpoint.hpp"
#include "foundation/ArgumentException.hpp"

namespace asmm = axis::services::messaging;
namespace afc = axis::foundation::collections;

asmm::CollectorEndpoint::CollectorEndpoint( void ) :
_eventListeners(axis::foundation::collections::ObjectSet::Create())
{
	// nothing to do here
}

asmm::CollectorEndpoint::~CollectorEndpoint( void )
{
	_eventListeners.Destroy();
}

void asmm::CollectorEndpoint::ConnectListener( MessageListener& listener )
{
	if (IsConnected(listener))
	{
		throw axis::foundation::ArgumentException(_T("listener"));
	}
	_eventListeners.Add(listener);
}

void asmm::CollectorEndpoint::DisconnectListener( MessageListener& listener )
{
	if (!IsConnected(listener))
	{
		throw axis::foundation::ArgumentException(_T("listener"));
	}
	_eventListeners.Remove(listener);
}

void asmm::CollectorEndpoint::DisconnectAll( void )
{
	_eventListeners.Clear();
}

bool asmm::CollectorEndpoint::IsConnected( MessageListener& listener ) const
{
	return _eventListeners.Contains(listener);
}

void asmm::CollectorEndpoint::DispatchMessage( Message& message ) const
{
  AddTracingInformation(message);
	for (afc::ObjectSet::Iterator it = _eventListeners.GetIterator(); it.HasNext(); it.GoNext())
	{
		// cast to correct type
		MessageListener& listener = (MessageListener&)it.GetItem();

		// request messaging processing
		listener.ProcessMessage(message);
	}
}

asmm::CollectorEndpoint::Iterator asmm::CollectorEndpoint::GetIterator( void ) const
{
	return _eventListeners.GetIterator();
}

void asmm::CollectorEndpoint::AddTracingInformation( asmm::Message& message ) const
{
  // nothing to do in base implementation
}
