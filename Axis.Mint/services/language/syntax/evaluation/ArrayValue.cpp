#include "ArrayValue.hpp"
#include "ArrayValueListImpl.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::ArrayValue::ArrayValue( void ) :
_items(*new ArrayValueListImpl())
{
	/* nothing more to do */
}

aslse::ArrayValue::~ArrayValue( void )
{
	delete &_items;
}

bool aslse::ArrayValue::IsAssignment( void ) const
{
	return false;
}

bool aslse::ArrayValue::IsAtomic( void ) const
{
	return false;
}

bool aslse::ArrayValue::IsNull( void ) const
{
	return false;
}

bool aslse::ArrayValue::IsArray( void ) const
{
	return true;
}

bool aslse::ArrayValue::IsEmpty( void ) const
{
	return _items.IsEmpty();
}

int aslse::ArrayValue::Count( void ) const
{
	return _items.Count();
}

aslse::ArrayValueList::Iterator aslse::ArrayValue::begin( void ) const
{
	return _items.begin();
}

aslse::ArrayValueList::Iterator aslse::ArrayValue::end( void ) const
{
	return _items.end();
}

aslse::ParameterValue& aslse::ArrayValue::Get( int pos ) const
{
	return _items.Get(pos);
}

aslse::ParameterValue& aslse::ArrayValue::operator[]( int pos ) const
{
	return Get(pos);
}

axis::String aslse::ArrayValue::ToString( void ) const
{
	axis::String str = _T("(");
	ArrayValueList::Iterator end = _items.end();
	for (ArrayValueList::Iterator it = _items.begin(); it != end; ++it)
	{
		str.append(it->ToString());
		if (it != end)
		{
			str.append(_T(","));
		}
	}
	str.append(_T(")"));
	return str;
}

aslse::ParameterValue& aslse::ArrayValue::Clone( void ) const
{
	ArrayValue *val = NULL;
	try
	{
		val = new ArrayValue();
		ArrayValueList::Iterator end = _items.end();
		for (ArrayValueList::Iterator it = _items.begin(); it != end; ++it)
		{
			val->AddValue((*it).Clone());
		}
	}
	catch (...)
	{
		if (val) delete val;
		throw;
	}
	return *val;
}

void aslse::ArrayValue::Clear( void )
{
	_items.Clear();
}

aslse::ArrayValue& aslse::ArrayValue::AddValue( ParameterValue& value )
{
	_items.AddValue(value);
	return *this;
}
