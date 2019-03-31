#include "SourceFileMetadata.hpp"


axis::services::messaging::metadata::SourceFileMetadata::SourceFileMetadata( const axis::String& sourceFileLocation, unsigned long lineIndex )
{
	_sourceFileLocation = sourceFileLocation;
	_lineIndex = lineIndex;
}

axis::services::messaging::metadata::SourceFileMetadata::~SourceFileMetadata( void )
{
	// nothing to do here
}

axis::services::messaging::metadata::Metadatum& axis::services::messaging::metadata::SourceFileMetadata::Clone( void ) const
{
	return *new SourceFileMetadata(*this);
}

void axis::services::messaging::metadata::SourceFileMetadata::Destroy( void ) const
{
	delete this;
}

axis::String axis::services::messaging::metadata::SourceFileMetadata::GetClassName( void )
{
	return _T("SourceFileMetadata.metadata.messaging.services.axis");
}

axis::String axis::services::messaging::metadata::SourceFileMetadata::GetTypeName( void ) const
{
	return GetClassName();
}

axis::String axis::services::messaging::metadata::SourceFileMetadata::GetSourceFileLocation( void ) const
{
	return _sourceFileLocation;
}

unsigned long axis::services::messaging::metadata::SourceFileMetadata::GetLineIndex( void ) const
{
	return _lineIndex;
}