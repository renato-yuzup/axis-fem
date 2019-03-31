#include "ErrorMessage.hpp"

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id ) :
EventMessage(id)
{
	_level = ErrorNormal;
	_exception = NULL;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, Severity level ) :
EventMessage(id)
{
	_level = level;
	_exception = NULL;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message ) :
EventMessage(id, message)
{
	_level = ErrorNormal;
	_exception = NULL;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, Severity level ) :
EventMessage(id, message)
{
	_level = level;
	_exception = NULL;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, const axis::String& title ) :
EventMessage(id, message, title)
{
	_level = ErrorNormal;
	_exception = NULL;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, const axis::String& title, Severity level ) :
EventMessage(id, message, title)
{
	_level = level;
	_exception = NULL;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::foundation::AxisException& e ) :
EventMessage(id)
{
	_level = ErrorNormal;
	_exception = &e;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::foundation::AxisException& e, Severity level ) :
EventMessage(id)
{
	_level = level;
	_exception = &e;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, const axis::foundation::AxisException& e ) :
EventMessage(id, message)
{
	_level = ErrorNormal;
	_exception = &e;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, const axis::foundation::AxisException& e, Severity level ) :
EventMessage(id, message)
{
	_level = level;
	_exception = &e;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, const axis::String& title, const axis::foundation::AxisException& e ) :
EventMessage(id, message, title)
{
	_level = ErrorNormal;
	_exception = &e;
}

axis::services::messaging::ErrorMessage::ErrorMessage( Message::id_type id, const axis::String& message, const axis::String& title, const axis::foundation::AxisException& e, Severity level ) :
EventMessage(id, message, title)
{
	_level = level;
	_exception = &e;
}
axis::services::messaging::ErrorMessage::~ErrorMessage( void )
{
	// nothing to do
}

bool axis::services::messaging::ErrorMessage::IsError( void ) const
{
	return true;
}

bool axis::services::messaging::ErrorMessage::IsWarning( void ) const
{
	return false;
}

bool axis::services::messaging::ErrorMessage::IsInfo( void ) const
{
	return false;
}

axis::services::messaging::ErrorMessage::Severity axis::services::messaging::ErrorMessage::GetSeverity( void ) const
{
	return _level;
}

void axis::services::messaging::ErrorMessage::DoDestroy( void ) const
{
	delete this;
}

axis::services::messaging::Message& axis::services::messaging::ErrorMessage::CloneMyself( id_type id ) const
{
	return *new ErrorMessage(id, GetDescription(), GetTitle(), _level);
}

bool axis::services::messaging::ErrorMessage::IsLogEntry( void ) const
{
	return false;
}

bool axis::services::messaging::ErrorMessage::HasAssociatedException( void ) const
{
	return _exception != NULL;
}

const axis::foundation::AxisException& axis::services::messaging::ErrorMessage::GetAssociatedException( void ) const
{
	return *_exception;
}