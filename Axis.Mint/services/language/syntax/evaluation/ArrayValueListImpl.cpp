#include "ArrayValueListImpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::ArrayValueListImpl::~ArrayValueListImpl( void )
{	// destroy the entire list
	Clear();
}

bool aslse::ArrayValueListImpl::IsEmpty( void ) const
{
	return _list.empty();
}

int aslse::ArrayValueListImpl::Count( void ) const
{
	return (int)_list.size();
}

aslse::ParameterValue& aslse::ArrayValueListImpl::Get( int pos ) const
{
	if (pos >= Count())
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return *_list[pos];
}

void aslse::ArrayValueListImpl::AddValue( ParameterValue& value )
{
	_list.push_back(&value);
}

void aslse::ArrayValueListImpl::Clear( void )
{	// this is a nothrow method -- so we must "eat" any exception thrown
	try
	{
		param_list::iterator end = _list.end();
		for (param_list::iterator it = _list.begin(); it != end; ++it)
		{
			// delete item
			delete *it;
		}

		// clear list
		_list.clear();
	}
	catch (...)
	{	// ignore it and force clear hash map
		try
		{
			_list.clear();
		}
		catch (...)
		{	// even so, we couldn't clear the list, simply ignore it
		}
	}
}

aslse::ArrayValueList::Iterator aslse::ArrayValueListImpl::end( void ) const
{
	return aslse::ArrayValueList::Iterator(IteratorLogicImpl(_list.end()));
}

aslse::ArrayValueList::Iterator aslse::ArrayValueListImpl::begin( void ) const
{
	return aslse::ArrayValueList::Iterator(IteratorLogicImpl(_list.begin()));
}

aslse::ArrayValueList& aslse::ArrayValueListImpl::Clone( void ) const
{
	ArrayValueListImpl *newList = new ArrayValueListImpl();
	try
	{
		// copy the entire list of items
		param_list::const_iterator end = _list.end();
		for (param_list::const_iterator it = _list.begin(); it != end; ++it)
		{
			newList->AddValue((*it)->Clone());
		}
	}
	catch(...)
	{
		throw;	/* let the calling function handle the exception */
	}
	return *newList;
}


/* ============================ BEGIN ITERATOR LOGIC IMPLEMENTATION ============================= */
aslse::ArrayValueListImpl::IteratorLogicImpl::IteratorLogicImpl( const param_list::iterator& it )
{
	_myIterator = it;
}

aslse::ArrayValueListImpl::IteratorLogicImpl::~IteratorLogicImpl( void )
{
	/* nothing to do here */
}

const aslse::ParameterValue& aslse::ArrayValueListImpl::IteratorLogicImpl::operator*( void ) const
{
	return **_myIterator;
}

const aslse::ParameterValue * aslse::ArrayValueListImpl::IteratorLogicImpl::operator->( void ) const
{
	return *_myIterator;
}

aslse::ArrayValueList::IteratorLogic& aslse::ArrayValueListImpl::IteratorLogicImpl::operator++( void )
{	// pre-fixed operation
	++_myIterator;
	return *this;
}

aslse::ArrayValueList::IteratorLogic& aslse::ArrayValueListImpl::IteratorLogicImpl::operator--( void )
{	// pre-fixed operation
	--_myIterator;
	return *this;
}

aslse::ArrayValueList::IteratorLogic& aslse::ArrayValueListImpl::IteratorLogicImpl::Clone( void ) const
{
	return *new IteratorLogicImpl(_myIterator);
}

bool aslse::ArrayValueListImpl::IteratorLogicImpl::operator==( const IteratorLogic& other ) const
{
	const IteratorLogicImpl& it = (const IteratorLogicImpl&)other;
	return _myIterator == it._myIterator;
}

void aslse::ArrayValueListImpl::IteratorLogicImpl::Destroy( void ) const
{
	delete this;
}
/* ============================= END ITERATOR LOGIC IMPLEMENTATION ============================== */
