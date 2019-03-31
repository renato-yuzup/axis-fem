#include "ArrayValueList.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::ArrayValueList::~ArrayValueList( void )
{
	/* Default destructor, nothing to do here */
}

aslse::ArrayValueList& aslse::ArrayValueList::operator=( const ArrayValueList& other )
{
	// clear the entire list
	Clear();

	// copy each item into our list
	Iterator end = other.end();
	for (Iterator it = other.begin(); it != end; ++it)
	{
		ParameterValue& newVal = it->Clone();
		AddValue(newVal);
	}

	return *this;
}


/* =============================== BEGIN ITERATOR IMPLEMENTATION ================================ */
aslse::ArrayValueList::Iterator::Iterator( const IteratorLogic& logic )
{
	// create a copy for us
	_logic = &logic.Clone();
}

aslse::ArrayValueList::Iterator::Iterator( const Iterator& it )
{
	_logic = NULL;
	Copy(it);
}

aslse::ArrayValueList::Iterator::Iterator( void )
{
	_logic = NULL;
}

const aslse::ParameterValue& aslse::ArrayValueList::Iterator::operator*( void ) const
{
	return **_logic;
}

const aslse::ParameterValue* aslse::ArrayValueList::Iterator::operator->( void ) const
{
	return &**_logic;
}

aslse::ArrayValueList::Iterator& aslse::ArrayValueList::Iterator::operator++( void )
{	// pre-fixed version
	++(*_logic);
	return *this;
}

aslse::ArrayValueList::Iterator aslse::ArrayValueList::Iterator::operator++( int )
{	 // post-fixed version
	aslse::ArrayValueList::Iterator it = *this;
	++(*_logic);
	return it;
}

aslse::ArrayValueList::Iterator& aslse::ArrayValueList::Iterator::operator--( void )
{	// pre-fixed version
	--(*_logic);
	return *this;
}

aslse::ArrayValueList::Iterator aslse::ArrayValueList::Iterator::operator--( int )
{	 // post-fixed version
	aslse::ArrayValueList::Iterator it = *this;
	--(*_logic);
	return it;
}

bool aslse::ArrayValueList::Iterator::operator==( const Iterator& it ) const
{
	return *_logic == *it._logic;
}

bool aslse::ArrayValueList::Iterator::operator!=( const Iterator& it ) const
{
	return !(*this == it);
}

aslse::ArrayValueList::Iterator& aslse::ArrayValueList::Iterator::operator=( const Iterator& it )
{
	Copy(it);
	return *this;
}

void aslse::ArrayValueList::Iterator::Copy( const Iterator& it )
{
	IteratorLogic &logic = it._logic->Clone();
	if (_logic != NULL) _logic->Destroy();
	_logic = &logic;
}
/* ================================= END ITERATOR IMPLEMENTATION ================================ */
