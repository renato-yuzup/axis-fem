#include "EventMessage.hpp"
#include "foundation/ArgumentException.hpp"

using namespace axis::foundation::collections;

axis::services::messaging::EventMessage::EventMessage( Message::id_type id ) :
Message(id), _tags(ObjectMap::Create())
{
}

axis::services::messaging::EventMessage::EventMessage( Message::id_type id, const axis::String& message ) :
Message(id), _tags(ObjectMap::Create())
{
	_description = message;
}

axis::services::messaging::EventMessage::EventMessage( Message::id_type id, const axis::String& message, const axis::String& title ) :
Message(id), _tags(ObjectMap::Create())
{
	_description = message;
	_title = title;
}

axis::services::messaging::EventMessage::~EventMessage( void )
{
	_tags.DestroyChildren();
	_tags.Destroy();
}

axis::String axis::services::messaging::EventMessage::GetDescription( void ) const
{
	return _description;
}

axis::String axis::services::messaging::EventMessage::GetTitle( void ) const
{
	return _title;
}

void axis::services::messaging::EventMessage::AppendTag( const axis::String& tagName, EventTag& tag )
{
	if (ContainsTag(tagName))
	{
		throw axis::foundation::ArgumentException(_T("tagName"));
	}
	_tags.Add(tagName, tag);
}

bool axis::services::messaging::EventMessage::ContainsTag( const axis::String& tagName ) const
{
	return _tags.Contains(tagName);
}

void axis::services::messaging::EventMessage::EraseTag( const axis::String& tagName )
{
	_tags.Remove(tagName);
}

void axis::services::messaging::EventMessage::ClearAllTags( void )
{
	_tags.Clear();
}

size_type axis::services::messaging::EventMessage::TagCount( void ) const
{
	return _tags.Count();
}

axis::services::messaging::Message& axis::services::messaging::EventMessage::DoClone( id_type id ) const
{
	EventMessage& clone = (EventMessage&)CloneMyself(id);
	for (ObjectMap::Iterator it = _tags.GetIterator(); it.HasNext(); it.GoNext())
	{
		clone.AppendTag(it.GetKey(), (EventTag&)it.GetItem());
	}
	return clone;
}

bool axis::services::messaging::EventMessage::IsEvent( void ) const
{
	return true;
}

bool axis::services::messaging::EventMessage::IsResult( void ) const
{
	return false;
}