#include "InputIterator.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace asli = axis::services::language::iterators;

asli::InputIterator::InputIterator( void )
{
	_iteratorImpl = NULL;
}

asli::InputIterator::InputIterator( const InputIterator& other )
{
	_iteratorImpl = &other._iteratorImpl->Clone();
	_iteratorImpl->NotifyUse();
}

asli::InputIterator::InputIterator( asli::IteratorLogic& logicImpl )
{
	_iteratorImpl = &logicImpl;
	_iteratorImpl->NotifyUse();
}

asli::InputIterator& asli::InputIterator::operator++( void )
{
	if (_iteratorImpl == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	++*_iteratorImpl;
	return *this;
}

asli::InputIterator asli::InputIterator::operator++( int )
{
	if (_iteratorImpl == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	iterators::IteratorLogic& logic = *_iteratorImpl++;
	return InputIterator(logic);
}

asli::InputIterator& asli::InputIterator::Clone( void ) const
{
	return *new InputIterator(*this);
}

const axis::String::value_type& asli::InputIterator::operator *( void ) const
{
	if (_iteratorImpl == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return **_iteratorImpl;
}

asli::InputIterator& asli::InputIterator::operator=( const InputIterator& it )
{
	if (&it == this)
	{	// ignore it
		return *this;
	}
	if (_iteratorImpl == it._iteratorImpl)
	{	// we are using the same iterator logic; ignore it
		return *this;
	}
	if (_iteratorImpl != NULL)
	{	// destroy previously used iterator logic
		_iteratorImpl->NotifyDestroy();
	}
	_iteratorImpl = &it._iteratorImpl->Clone();
	_iteratorImpl->NotifyUse();

	return *this;
}

bool asli::InputIterator::operator!=( const InputIterator& it ) const
{
	return !(*this == it);
}

bool asli::InputIterator::operator==( const InputIterator& it ) const
{
	if (_iteratorImpl == NULL)	// an invalid iterator is not equal to any other iterator
	{
		return false;
	}
	return *_iteratorImpl == *it._iteratorImpl;
}

asli::InputIterator::~InputIterator( void )
{
	if (_iteratorImpl != NULL)
	{
		_iteratorImpl->NotifyDestroy();
	}
}

axis::String asli::InputIterator::ToString( const asli::InputIterator& end ) const
{
	axis::String result;
	axis::String::value_type str[2];
	str[1] = NULL;
	for (InputIterator it = *this; it != end; ++it)
	{
		str[0] = *it;
		result.append(str);
	}
	return result;
}

bool asli::InputIterator::operator>( const InputIterator& other ) const
{
	return *_iteratorImpl > *other._iteratorImpl;
}

bool asli::InputIterator::operator<( const InputIterator& other ) const
{
	return !(*this >= other);
}

bool asli::InputIterator::operator>=( const InputIterator& other ) const
{
	return (*this > other) || (*this == other);
}

bool asli::InputIterator::operator<=( const InputIterator& other ) const
{
	return !(*this > other);
}
