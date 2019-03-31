#include "SourceHintSet.hpp"
#include "ArgumentException.hpp"

void axis::foundation::SourceHintSet::Add( const SourceTraceHint& hint )
{
	if (Contains(hint))
	{
		throw axis::foundation::ArgumentException();
	}
	_items.insert(&hint);	
}

void axis::foundation::SourceHintSet::Remove( const SourceTraceHint& hint )
{
	if (!Contains(hint))
	{
		throw axis::foundation::ArgumentException();
	}
	_items.erase(&hint);
}

void axis::foundation::SourceHintSet::Clear( void )
{
	_items.clear();
}

bool axis::foundation::SourceHintSet::IsEmpty( void ) const
{
	return _items.size() == 0;
}

bool axis::foundation::SourceHintSet::Contains( const SourceTraceHint& hint ) const
{
	return _items.find(&hint) != _items.cend();
}

axis::foundation::SourceHintSet::Visitor& axis::foundation::SourceHintSet::GetVisitor( void ) const
{
	return *new SourceHintSetVisitor(_items.begin(), _items.end(), _items.begin());
}




axis::foundation::SourceHintSet::SourceHintSetVisitor::SourceHintSetVisitor( hint_set::iterator begin, hint_set::iterator end, hint_set::iterator current )
{
	_begin = begin; _end = end; _current = current;
}

const axis::foundation::SourceTraceHint& axis::foundation::SourceHintSet::SourceHintSetVisitor::GetItem( void ) const
{
	return **_current;
}

bool axis::foundation::SourceHintSet::SourceHintSetVisitor::HasNext( void ) const
{
	return _current != _end;
}

void axis::foundation::SourceHintSet::SourceHintSetVisitor::GoNext( void )
{
	++_current;
}

void axis::foundation::SourceHintSet::SourceHintSetVisitor::Reset( void )
{
	_current = _begin;
}

axis::foundation::SourceHintSet::SourceHintSetVisitor::~SourceHintSetVisitor( void )
{
	// nothing to do
}