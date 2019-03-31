#include "ParameterList.hpp"
#include "ParameterListImpl.hpp"
#include "ParameterAssignment.hpp"
#include "foundation/ArgumentException.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

const aslse::ParameterList& aslse::ParameterList::Empty = *new ParameterListImpl();

aslse::ParameterList::~ParameterList( void )
{
	/* Default destructor, nothing to do here */
}

aslse::ParameterList& aslse::ParameterList::operator=( const ParameterList& other )
{
	// clear the entire list
	Clear();

	// copy each item into our list
	Iterator end = other.end();
	for (Iterator it = other.begin(); it != end; ++it)
	{
		ParameterValue& newVal = it->Value->Clone();
		AddParameter(it->Name, newVal);
	}

	return *this;
}

aslse::ParameterList& aslse::ParameterList::Create( void )
{
	return *new ParameterListImpl();
}

aslse::ParameterList& aslse::ParameterList::FromParameterArray( const aslse::ArrayValue& arrayValue )
{
	ParameterList& paramList = Create();

	// iterate through each array value
	size_type count = arrayValue.Count();
	for (size_type i = 0; i < count; ++i)
	{
		ParameterValue& value = arrayValue.Get(0);
		if (value.IsAssignment())
		{
			ParameterAssignment& assignment = static_cast<ParameterAssignment&>(value);
			String paramName = assignment.GetIdName();
			ParameterValue& paramValue = assignment.GetRhsValue().Clone();
			paramList.AddParameter(paramName, paramValue);
		}
		else
		{	// what? not an assignment? it is then an invalid array to parse
			paramList.Destroy();
			throw axis::foundation::ArgumentException(_T("Invalid parameter array."));
		}
	}

	return paramList;
}

/* ================================= BEGIN PAIR IMPLEMENTATION ================================== */
aslse::ParameterList::Pair::Pair( const axis::String& name, ParameterValue *value ) :
Name(name), Value(value)
{
	/* nothing more to do here */
}

aslse::ParameterList::Pair::Pair( void ) :
Name(), Value(NULL)
{
	/* nothing more to do here */
}

aslse::ParameterList::Pair::Pair( const Pair& other ) :
Name(other.Name), Value(other.Value)
{
	/* nothing more to do here */
}

aslse::ParameterList::Pair& aslse::ParameterList::Pair::operator=( const Pair& pair )
{
	Name = pair.Name;
	Value = pair.Value;
	return *this;
}

bool aslse::ParameterList::Pair::operator==( const Pair& pair ) const
{
	return (Name.compare(pair.Name) == 0 &&
			&Value == &pair.Value);
}

bool aslse::ParameterList::Pair::operator!=( const Pair& pair ) const
{
	return !(*this == pair);
}
/* ================================== END PAIR IMPLEMENTATION =================================== */


/* =============================== BEGIN ITERATOR IMPLEMENTATION ================================ */
aslse::ParameterList::Iterator::Iterator( const IteratorLogic& logic )
{
	// create a copy for us
	_logic = &logic.Clone();
}

aslse::ParameterList::Iterator::Iterator( const Iterator& it )
{
	_logic = NULL;
	Copy(it);
}

aslse::ParameterList::Iterator::Iterator( void )
{
	_logic = NULL;
}

const aslse::ParameterList::Pair& aslse::ParameterList::Iterator::operator*( void ) const
{
	return **_logic;
}

const aslse::ParameterList::Pair* aslse::ParameterList::Iterator::operator->( void ) const
{
	return &**_logic;
}

aslse::ParameterList::Iterator& aslse::ParameterList::Iterator::operator++( void )
{	// pre-fixed version
	++(*_logic);
	return *this;
}

aslse::ParameterList::Iterator aslse::ParameterList::Iterator::operator++( int )
{	 // post-fixed version
	aslse::ParameterList::Iterator it = *this;
	++(*_logic);
	return it;
}

aslse::ParameterList::Iterator& aslse::ParameterList::Iterator::operator--( void )
{	// pre-fixed version
	--(*_logic);
	return *this;
}

aslse::ParameterList::Iterator aslse::ParameterList::Iterator::operator--( int )
{	 // post-fixed version
	aslse::ParameterList::Iterator it = *this;
	--(*_logic);
	return it;
}

bool aslse::ParameterList::Iterator::operator==( const Iterator& it ) const
{
	return (*_logic) == (*it._logic);
}

bool aslse::ParameterList::Iterator::operator!=( const Iterator& it ) const
{
	return !((*this) == it);
}

aslse::ParameterList::Iterator& aslse::ParameterList::Iterator::operator=( const Iterator& it )
{
	Copy(it);
	return *this;
}

void aslse::ParameterList::Iterator::Copy( const Iterator& it )
{
	IteratorLogic &logic = it._logic->Clone();
	if (_logic != NULL) delete _logic;
	_logic = &logic;
}
/* ================================ END ITERATOR IMPLEMENTATION ================================= */
