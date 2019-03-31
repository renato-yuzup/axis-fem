#include "ComponentMetadata.hpp"

axis::services::configuration::ComponentMetadata::ComponentMetadata( const char *featurePath, const char *shortName )
{
	CopyStr(&_featurePath, featurePath);
	CopyStr(&_featureName, shortName);
}

axis::services::configuration::ComponentMetadata::~ComponentMetadata( void )
{
	delete _featureName;
	delete _featurePath;
}

void axis::services::configuration::ComponentMetadata::CopyStr( char **ptr, const char *source )
{
	size_t count = 0;

	// count how many characters
	const char *p = source;
	while(*(p++) != NULL) count++;

	// allocate new buffer
	*ptr = new char[count + 1];
	(*ptr)[count] = NULL;

	// copy string
	for (size_t i = 0; i < count; i++)
	{
		(*ptr)[i] = source[i];
	}
}

const char * axis::services::configuration::ComponentMetadata::FeaturePath( void ) const
{
	return _featurePath;
}

const char * axis::services::configuration::ComponentMetadata::ShortName( void ) const
{
	return _featureName;
}