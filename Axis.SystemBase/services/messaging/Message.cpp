#include "Message.hpp"
#include "foundation/NotSupportedException.hpp"

axis::services::messaging::Message& axis::services::messaging::Message::operator=( const Message& message )
{
	throw axis::foundation::NotSupportedException();
}

axis::services::messaging::Message::Message( const Message& message )
{
	throw axis::foundation::NotSupportedException();
}

axis::services::messaging::Message::Message( id_type id )
{
	_id = id;
	_timestamp = axis::foundation::date_time::Timestamp::GetLocalTime();
	_metadata = &axis::services::messaging::metadata::MetadataCollection::Create();
}

axis::services::messaging::Message::~Message( void )
{
	delete _metadata;
}

axis::services::messaging::Message::id_type axis::services::messaging::Message::GetId( void ) const
{
	return _id;
}

axis::foundation::date_time::Timestamp axis::services::messaging::Message::GetTimestamp( void ) const
{
	return _timestamp;
}

const axis::services::messaging::TraceInfoCollection& axis::services::messaging::Message::GetTraceInformation( void ) const
{
	return _traceInformation;
}

axis::services::messaging::TraceInfoCollection& axis::services::messaging::Message::GetTraceInformation( void )
{
	return _traceInformation;
}

const axis::services::messaging::metadata::MetadataCollection& axis::services::messaging::Message::GetMetadata( void ) const
{
	return *_metadata;
}

axis::services::messaging::metadata::MetadataCollection& axis::services::messaging::Message::GetMetadata( void )
{
	return *_metadata;
}

axis::services::messaging::Message::self& axis::services::messaging::Message::operator<<( const TraceInfo& traceInfo )
{
	_traceInformation.AddTraceInfo(traceInfo);
	return *this;
}

void axis::services::messaging::Message::Destroy( void ) const
{
	// up to this code version, no destruction procedures is
	// required in the base class; just forward control to child
	// classes
	DoDestroy();
}

axis::services::messaging::Message& axis::services::messaging::Message::Clone( void ) const
{
	// create a new instance of this abstract class
	Message& msg = DoClone(_id);
	msg._id = _id;
	msg._timestamp = _timestamp;
	msg._metadata->Destroy();
	msg._metadata = &_metadata->Clone();

	// copy information stack
	msg._traceInformation = _traceInformation;

	return msg;
}

