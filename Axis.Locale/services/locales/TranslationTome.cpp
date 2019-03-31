#include "stdafx.h"
#include "TranslationTome.hpp"
#include <assert.h>

namespace {
	void EnforceRangePolicy(long minId, long maxId)
	{
		assert((minId > 0 && maxId > 0) && "Translation tome bad behavior: cannot have non-positive bounds in translation range.");
		assert((minId <= maxId) && "Translation tome bad behavior: invalid translation range.");
	}
}

axis::services::locales::TranslationTome::~TranslationTome( void )
{
	// nothing to do here
}

axis::String axis::services::locales::TranslationTome::Translate( long messageId ) const
{
	if (CanTranslate(messageId))
	{
		return DoTranslate(messageId);
	}

	return _T("<message description not available>");
}

bool axis::services::locales::TranslationTome::CanTranslate( long messageId ) const
{
	// if id is out of range, it automatically cannot be translated
	long minId = DoGetFirstTranslationIdRange();
	long maxId = DoGetLastTranslationIdRange();
	EnforceRangePolicy(minId, maxId);
	if (messageId >= minId && messageId <= maxId)
	{
		return DoCanTranslate(messageId);
	}

	return false;
}

size_type axis::services::locales::TranslationTome::GetEntryCount( void ) const
{
	size_type maxEntryCount = (size_type)(DoGetLastTranslationIdRange() - DoGetFirstTranslationIdRange() + 1);
	size_type count = DoGetEntryCount();
	if (count > maxEntryCount)	// bad behavior...
	{
		assert(!"Translation tome bad behavior: entry count larger than expected.");
	}
	return count;
}

