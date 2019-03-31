#include "PluginLibrarian.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace asmg = axis::services::management;

asmg::PluginLibrarian::~PluginLibrarian( void )
{
	// nothing to do here
}

asmg::PluginLibrarian::Iterator::Iterator( void )
{
	_current = NULL;
	_end = NULL;
}

asmg::PluginLibrarian::Iterator::Iterator( const IteratorLogic& begin, const IteratorLogic& end )
{
	_current = &begin.Clone();
	_end = &end.Clone();
}

asmg::PluginLibrarian::Iterator::Iterator( const Iterator& it )
{
	_current = &it._current->Clone();
	_end = &it._end->Clone();
}

asmg::PluginLibrarian::Iterator::~Iterator( void )
{
	_current->Destroy();
	_end->Destroy();
	_current = NULL;
	_end = NULL;
}

asmg::PluginLibrarian::Iterator& asmg::PluginLibrarian::Iterator::operator=( const Iterator& it )
{
	// ignore self-assignment
	if (this == &it) return *this;

	// create new iterators
	IteratorLogic *current = NULL, *end = NULL;
	try
	{
		current = &it._current->Clone();
		end = &it._end->Clone();
	}
	catch (...)
	{
		if (current) current->Destroy();
		if (end) end->Destroy();
		throw;
	}

	// delete old iterators
	if (_current) _current->Destroy();
	if (_end) _end->Destroy();

	_current = current;
	_end = end;

	return *this;
}

bool asmg::PluginLibrarian::Iterator::operator==( const Iterator& it ) const
{
	return *_current == *it._current;
}

bool asmg::PluginLibrarian::Iterator::operator!=( const Iterator& it ) const
{
	return !(*this == it);
}

bool asmg::PluginLibrarian::Iterator::HasNext( void ) const
{
	if (_current == NULL) return false;
	return *_current != *_end;
}

asmg::PluginLibrarian::Iterator& asmg::PluginLibrarian::Iterator::GoNext( void )
{
	if (!HasNext()) throw axis::foundation::InvalidOperationException();
	_current->GoNext();
	return *this;
}

asmg::PluginConnector& asmg::PluginLibrarian::Iterator::GetItem( void ) const
{
	if (_current == NULL) throw axis::foundation::InvalidOperationException();
	return _current->GetItem();
}

asmg::PluginConnector * asmg::PluginLibrarian::Iterator::operator->( void ) const
{
	return &_current->GetItem();
}

asmg::PluginConnector& asmg::PluginLibrarian::Iterator::operator*( void ) const
{
	return _current->GetItem();
}
