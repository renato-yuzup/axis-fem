#include "SourceTraceHint.hpp"


axis::foundation::SourceTraceHint::SourceTraceHint(void)
{
	_hintId = 0;
}

axis::foundation::SourceTraceHint::SourceTraceHint( int hintId )
{
	_hintId = hintId;
}

axis::foundation::SourceTraceHint::~SourceTraceHint(void)
{
	// nothing to do
}

bool axis::foundation::SourceTraceHint::operator==( const SourceTraceHint& sth ) const
{
	return _hintId == sth._hintId;
}

bool axis::foundation::SourceTraceHint::operator!=( const SourceTraceHint& sth ) const
{
	return _hintId != sth._hintId;
}

bool axis::foundation::SourceTraceHint::operator>=( const SourceTraceHint& sth ) const
{
	return _hintId >= sth._hintId;
}

bool axis::foundation::SourceTraceHint::operator<=( const SourceTraceHint& sth ) const
{
	return _hintId <= sth._hintId;
}

bool axis::foundation::SourceTraceHint::operator>( const SourceTraceHint& sth ) const
{
	return _hintId > sth._hintId;
}

bool axis::foundation::SourceTraceHint::operator<( const SourceTraceHint& sth ) const
{
	return _hintId < sth._hintId;
}