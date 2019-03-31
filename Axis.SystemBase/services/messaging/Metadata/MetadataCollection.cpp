#include "MetadataCollection.hpp"


axis::services::messaging::metadata::MetadataCollection::MetadataCollection( void ) :
_metadata(axis::foundation::collections::ObjectMap::Create())
{
	// nothing to do here
}

axis::services::messaging::metadata::MetadataCollection::~MetadataCollection( void )
{
	_metadata.DestroyChildren();
	delete &_metadata;
}

axis::services::messaging::metadata::MetadataCollection& axis::services::messaging::metadata::MetadataCollection::Create( void )
{
	return *new MetadataCollection();
}

void axis::services::messaging::metadata::MetadataCollection::Add( const Metadatum& metadatum )
{
	_metadata.Add(metadatum.GetTypeName(), metadatum.Clone());
}

bool axis::services::messaging::metadata::MetadataCollection::Contains( const axis::String& typeName ) const
{
	return _metadata.Contains(typeName);
}

void axis::services::messaging::metadata::MetadataCollection::Remove( const axis::String& typeName )
{
	Metadatum& m = static_cast<Metadatum&>(_metadata.Get(typeName));
	_metadata.Remove(typeName);
	m.Destroy();
}

axis::services::messaging::metadata::Metadatum& axis::services::messaging::metadata::MetadataCollection::Get( const axis::String& typeName ) const
{
	return static_cast<Metadatum&>(_metadata.Get(typeName));
}

axis::services::messaging::metadata::Metadatum& axis::services::messaging::metadata::MetadataCollection::operator[]( const axis::String& typeName ) const
{
	return static_cast<Metadatum&>(_metadata.Get(typeName));
}

void axis::services::messaging::metadata::MetadataCollection::Clear( void )
{
	_metadata.DestroyChildren();	
}

size_type axis::services::messaging::metadata::MetadataCollection::Count( void ) const
{
	return _metadata.Count();
}

bool axis::services::messaging::metadata::MetadataCollection::Empty( void ) const
{
	return _metadata.Count() == 0;
}

axis::services::messaging::metadata::MetadataCollection& axis::services::messaging::metadata::MetadataCollection::Clone( void ) const
{
	// first, create a new collection
	MetadataCollection& c = *new MetadataCollection();

	// clone each children (the add operation already clones the
	// object)
	for (axis::foundation::collections::ObjectMap::Iterator it = _metadata.GetIterator(); it.HasNext(); it.GoNext())
	{
		c.Add(static_cast<Metadatum&>(it.GetItem()));
	}

	return c;
}

void axis::services::messaging::metadata::MetadataCollection::Destroy( void ) const
{
	delete this;
}