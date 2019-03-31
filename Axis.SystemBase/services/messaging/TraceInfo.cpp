#include "TraceInfo.hpp"
#include "foundation/InvalidOperationException.hpp"

axis::services::messaging::TraceInfo::TraceInfo( int sourceId )
{
	_sourceId = sourceId;
	_sourceTag = NULL;
}

axis::services::messaging::TraceInfo::TraceInfo( const axis::String& sourceName )
{
	_sourceId = 0;
	_sourceName = sourceName;
	_sourceTag = NULL;
}

axis::services::messaging::TraceInfo::TraceInfo( int sourceId, const axis::String& sourceName )
{
	_sourceId = sourceId;
	_sourceName = sourceName;
	_sourceTag = NULL;
}

axis::services::messaging::TraceInfo::TraceInfo( int sourceId, const axis::String& sourceName, const TraceTag& tag )
{
	_sourceId = sourceId;
	_sourceName = sourceName;
	_sourceTag = &tag.Clone();
}

axis::services::messaging::TraceInfo::TraceInfo( int sourceId, const TraceTag& tag )
{
	_sourceId = sourceId;
	_sourceTag = &tag.Clone();
}

axis::services::messaging::TraceInfo::TraceInfo( const TraceInfo& other )
{
	_sourceTag = NULL;
	Copy(other);
}

axis::services::messaging::TraceInfo::~TraceInfo( void )
{
	if (_sourceTag != NULL) _sourceTag->Destroy();
}

axis::String& axis::services::messaging::TraceInfo::SourceName( void )
{
	return _sourceName;
}

const axis::String& axis::services::messaging::TraceInfo::SourceName( void ) const
{
	return _sourceName;
}

int& axis::services::messaging::TraceInfo::SourceId( void )
{
	return _sourceId;
}

const int& axis::services::messaging::TraceInfo::SourceId( void ) const
{
	return _sourceId;
}

void axis::services::messaging::TraceInfo::AppendTag( const TraceTag& tag )
{
	if (_sourceTag != NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	_sourceTag = &tag.Clone();
}

void axis::services::messaging::TraceInfo::ReplaceTag( const TraceTag& tag )
{
	TraceTag& newTag = tag.Clone();
	if (_sourceTag != NULL) _sourceTag->Destroy();
	_sourceTag = &newTag;
}

void axis::services::messaging::TraceInfo::EraseTag( void )
{
	if (_sourceTag != NULL) _sourceTag->Destroy();
	_sourceTag = NULL;
}

bool axis::services::messaging::TraceInfo::IsTagged( void ) const
{
	return _sourceTag != NULL;
}

axis::services::messaging::TraceInfo& axis::services::messaging::TraceInfo::operator=( const TraceInfo& other )
{
	Copy(other);
	return *this;
}

void axis::services::messaging::TraceInfo::Copy( const TraceInfo& other )
{
	axis::services::messaging::TraceTag *newTag = NULL;
	if (other._sourceTag != NULL)
	{
		newTag = &other._sourceTag->Clone();
	}
	
	// copy informations
	_sourceId = other._sourceId;
	_sourceName = other._sourceName;
	if (_sourceTag != NULL) _sourceTag->Destroy();
	_sourceTag = newTag;
}
