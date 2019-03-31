#include "stdafx.h"
#include "TranslationFacet.hpp"
#include "foundation/ArgumentException.hpp"

#include <set>

class axis::services::locales::TranslationFacet::TranslationData
{
public:
	typedef std::set<const TranslationTome *> tome_set;
	tome_set Tomes;
};

axis::services::locales::TranslationFacet::TranslationFacet( void )
{
	_data = new TranslationData();
}

axis::services::locales::TranslationFacet::~TranslationFacet( void )
{
	delete _data;
}

axis::String axis::services::locales::TranslationFacet::Translate( long messageId ) const
{
	TranslationData::tome_set::iterator end = _data->Tomes.end();
	for (TranslationData::tome_set::iterator it = _data->Tomes.begin(); it != end; ++it)
	{
		const TranslationTome& tome = **it;
		if (tome.CanTranslate(messageId))
		{
			return tome.Translate(messageId);
		}
	}
	return _T("");
}

bool axis::services::locales::TranslationFacet::CanTranslate( long messageId ) const
{
	TranslationData::tome_set::iterator end = _data->Tomes.end();
	for (TranslationData::tome_set::iterator it = _data->Tomes.begin(); it != end; ++it)
	{
		const TranslationTome& tome = **it;
		if (tome.CanTranslate(messageId))
		{
			return true;
		}
	}
	return false;
}

void axis::services::locales::TranslationFacet::AddTome( const axis::services::locales::TranslationTome& tome )
{
	if (_data->Tomes.find(&tome) != _data->Tomes.end())
	{
		throw axis::foundation::ArgumentException(_T("tome"));
	}
	_data->Tomes.insert(&tome);
}

size_type axis::services::locales::TranslationFacet::GetTomeCount( void ) const
{
	return (size_type)_data->Tomes.size();
}

size_type axis::services::locales::TranslationFacet::GetTotalEntriesCount( void ) const
{
	size_type count = 0;
	TranslationData::tome_set::iterator end = _data->Tomes.end();
	for (TranslationData::tome_set::iterator it = _data->Tomes.begin(); it != end; ++it)
	{
		count += (*it)->GetEntryCount();
	}
	return count;
}