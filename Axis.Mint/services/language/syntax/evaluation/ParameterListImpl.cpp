#include "ParameterListImpl.hpp"
#include "foundation/ArgumentException.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::ParameterListImpl::~ParameterListImpl( void )
{	// destroy the entire list
	Clear();
}

bool aslse::ParameterListImpl::IsEmpty( void ) const
{
	return _list.empty();
}

int aslse::ParameterListImpl::Count( void ) const
{
	return (int)_list.size();
}

aslse::ParameterValue& aslse::ParameterListImpl::GetParameterValue( const axis::String& name ) const
{
	param_list::const_iterator it = _list.find(name);
	if(it == _list.end()) throw axis::foundation::ArgumentException();
	return *it->second;
}

bool aslse::ParameterListImpl::IsDeclared( const axis::String& name ) const
{
	return (_list.find(name) != _list.end());
}

aslse::ParameterList& aslse::ParameterListImpl::Consume( const axis::String& name )
{
	if (!IsDeclared(name))
	{
		throw axis::foundation::ArgumentException();
	}
	_list.erase(name);
	return *this;
}

aslse::ParameterList& aslse::ParameterListImpl::AddParameter( const axis::String& name, 
                                                              ParameterValue& value )
{
	if (IsDeclared(name))
	{
		throw axis::foundation::ArgumentException();
	}
	_list[name] = &value;
	return *this;
}

void aslse::ParameterListImpl::Clear( void )
{	// this is a nothrow method -- so we must "eat" any exception thrown
	try
	{
		param_list::iterator end = _list.end();
		for (param_list::iterator it = _list.begin(); it != end; ++it)
		{
			// delete item
			delete it->second;
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

aslse::ParameterList::Iterator aslse::ParameterListImpl::end( void ) const
{
	return aslse::ParameterList::Iterator(IteratorLogicImpl(_list.end()));
}

aslse::ParameterList::Iterator aslse::ParameterListImpl::begin( void ) const
{
	return aslse::ParameterList::Iterator(IteratorLogicImpl(_list.begin()));
}

aslse::ParameterList& aslse::ParameterListImpl::Clone( void ) const
{
	ParameterListImpl *newList = new ParameterListImpl();
	try
	{
		// copy the entire list of items
		param_list::const_iterator end = _list.end();
		for (param_list::const_iterator it = _list.begin(); it != end; ++it)
		{
			newList->AddParameter(it->first, it->second->Clone());
		}
	}
	catch(...)
	{
		throw;	/* let the calling function handle the exception */
	}
	return *newList;
}

void aslse::ParameterListImpl::Destroy( void ) const
{
	delete this;
}

/* ============================ BEGIN ITERATOR LOGIC IMPLEMENTATION ============================= */
aslse::ParameterListImpl::IteratorLogicImpl::IteratorLogicImpl( param_list::iterator it )
{
	_myIterator = it;
	_isPairValid = false;
	_myPair = NULL;
}

aslse::ParameterListImpl::IteratorLogicImpl::~IteratorLogicImpl( void )
{
	/* nothing to do here */
}

const aslse::ParameterList::Pair& aslse::ParameterListImpl::IteratorLogicImpl::operator*( void ) const
{
	if (!_isPairValid)
	{
		if (_myPair != NULL) delete _myPair;
		_myPair = new ParameterList::Pair(_myIterator->first, _myIterator->second);
		_isPairValid = true;
	}
	return *_myPair;
}

const aslse::ParameterList::Pair * aslse::ParameterListImpl::IteratorLogicImpl::operator->( void ) const
{
	if (!_isPairValid)
	{
		if (_myPair != NULL) delete _myPair;
		_myPair = new ParameterList::Pair(_myIterator->first, _myIterator->second);
		_isPairValid = true;
	}
	return _myPair;
}

aslse::ParameterList::IteratorLogic& aslse::ParameterListImpl::IteratorLogicImpl::operator++( void )
{	// pre-fixed operation
	++_myIterator;
	_isPairValid = false;
	return *this;
}

aslse::ParameterList::IteratorLogic& aslse::ParameterListImpl::IteratorLogicImpl::operator--( void )
{	// pre-fixed operation
	--_myIterator;
	_isPairValid = false;
	return *this;
}

aslse::ParameterList::IteratorLogic& aslse::ParameterListImpl::IteratorLogicImpl::Clone( void ) const
{
	return *new IteratorLogicImpl(_myIterator);
}

bool aslse::ParameterListImpl::IteratorLogicImpl::operator==( const IteratorLogic& other ) const
{
	const IteratorLogicImpl& it = (const IteratorLogicImpl&)other;
	return _myIterator == it._myIterator;
}
/* ============================= END ITERATOR LOGIC IMPLEMENTATION ============================== */
