#include "ResultMessage.hpp"

axis::services::messaging::ResultMessage::ResultMessage( Message::id_type id ) :
Message(id)
{
	// nothing to do here
}

axis::services::messaging::ResultMessage::ResultMessage( Message::id_type id, const axis::String& description ) :
Message(id)
{
	_description = description;
}

axis::services::messaging::ResultMessage::~ResultMessage( void )
{
	// nothing to do here
}

axis::String axis::services::messaging::ResultMessage::GetDescription( void ) const
{
	return _description;
}

bool axis::services::messaging::ResultMessage::IsEvent( void ) const
{
	return false;
}

bool axis::services::messaging::ResultMessage::IsResult( void ) const
{
	return true;
}