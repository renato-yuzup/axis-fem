#include "WarningMessage.hpp"

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id ) :
EventMessage(id)
{
	_level = WarningNormal;
	_exception = NULL;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, Severity level ) :
EventMessage(id)
{
	_level = level;
	_exception = NULL;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message ) :
EventMessage(id, message)
{
	_level = WarningNormal;
	_exception = NULL;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, Severity level ) :
EventMessage(id, message)
{
	_level = level;
	_exception = NULL;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, const axis::String& title ) :
EventMessage(id, message, title)
{
	_level = WarningNormal;
	_exception = NULL;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, const axis::String& title, Severity level ) :
EventMessage(id, message, title)
{
	_level = level;
	_exception = NULL;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::foundation::AxisException& e ) :
EventMessage(id)
{
	_level = WarningNormal;
	_exception = &e;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::foundation::AxisException& e, Severity level ) :
EventMessage(id)
{
	_level = level;
	_exception = &e;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, const axis::foundation::AxisException& e ) :
EventMessage(id, message)
{
	_level = WarningNormal;
	_exception = &e;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, const axis::foundation::AxisException& e, Severity level ) :
EventMessage(id, message)
{
	_level = level;
	_exception = &e;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, const axis::String& title, const axis::foundation::AxisException& e ) :
EventMessage(id, message, title)
{
	_level = WarningNormal;
	_exception = &e;
}

axis::services::messaging::WarningMessage::WarningMessage( Message::id_type id, const axis::String& message, const axis::String& title, const axis::foundation::AxisException& e, Severity level ) :
EventMessage(id, message, title)
{
	_level = level;
	_exception = &e;
}
axis::services::messaging::WarningMessage::~WarningMessage( void )
{
	// nothing to do
}

axis::services::messaging::WarningMessage::Severity axis::services::messaging::WarningMessage::GetSeverity( void ) const
{
	return _level;
}

bool axis::services::messaging::WarningMessage::IsError( void ) const
{
	return false;
}

bool axis::services::messaging::WarningMessage::IsWarning( void ) const
{
	return true;
}

bool axis::services::messaging::WarningMessage::IsInfo( void ) const
{
	return false;
}

void axis::services::messaging::WarningMessage::DoDestroy( void ) const
{
	delete this;
}

axis::services::messaging::Message& axis::services::messaging::WarningMessage::CloneMyself( id_type id ) const
{
	return *new WarningMessage(id, GetDescription(), GetTitle(), _level);
}

bool axis::services::messaging::WarningMessage::IsLogEntry( void ) const
{
	return false;
}

bool axis::services::messaging::WarningMessage::HasAssociatedException( void ) const
{
	return _exception != NULL;
}

const axis::foundation::AxisException& axis::services::messaging::WarningMessage::GetAssociatedException( void ) const
{
	return *_exception;
}