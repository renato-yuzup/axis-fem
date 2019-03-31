#include "InfoMessage.hpp"

axis::services::messaging::InfoMessage::InfoMessage( Message::id_type id ) :
EventMessage(id)
{
	_level = InfoNormal;
}

axis::services::messaging::InfoMessage::InfoMessage( Message::id_type id, InfoLevel level ) :
EventMessage(id)
{
	_level = level;
}

axis::services::messaging::InfoMessage::InfoMessage( Message::id_type id, const axis::String& message ) :
EventMessage(id, message)
{
	_level = InfoNormal;
}

axis::services::messaging::InfoMessage::InfoMessage( Message::id_type id, const axis::String& message, InfoLevel level ) :
EventMessage(id, message)
{
	_level = level;
}

axis::services::messaging::InfoMessage::InfoMessage( Message::id_type id, const axis::String& message, const axis::String& title ) :
EventMessage(id, message, title)
{
	_level = InfoNormal;
}

axis::services::messaging::InfoMessage::InfoMessage( Message::id_type id, const axis::String& message, const axis::String& title, InfoLevel level ) :
EventMessage(id, message, title)
{
	_level = level;
}

axis::services::messaging::InfoMessage::~InfoMessage( void )
{
	// nothing to do
}

bool axis::services::messaging::InfoMessage::IsError( void ) const
{
	return false;
}

bool axis::services::messaging::InfoMessage::IsWarning( void ) const
{
	return false;
}

bool axis::services::messaging::InfoMessage::IsInfo( void ) const
{
	return true;
}

axis::services::messaging::InfoMessage::InfoLevel axis::services::messaging::InfoMessage::GetInfoLevel( void ) const
{
	return _level;
}

void axis::services::messaging::InfoMessage::DoDestroy( void ) const
{
	delete this;
}

axis::services::messaging::Message& axis::services::messaging::InfoMessage::CloneMyself( id_type _id ) const
{
	return *new InfoMessage(_id, GetDescription(), GetTitle(), _level);
}

bool axis::services::messaging::InfoMessage::IsLogEntry( void ) const
{
	return false;
}
