#include "StringIteratorLogic.hpp"

namespace asli = axis::services::language::iterators;

asli::StringIteratorLogic::StringIteratorLogic( const StringIteratorLogic& it )
{
	_it = it._it;
}

asli::StringIteratorLogic::StringIteratorLogic( const axis::String& sourceStr )
{
	_it = sourceStr. cbegin();
}

asli::StringIteratorLogic::StringIteratorLogic( const axis::String::const_iterator& sourceIt )
{
	_it = sourceIt;
}

asli::StringIteratorLogic::~StringIteratorLogic( void )
{
	/* Nothing to do */
}

asli::IteratorLogic& asli::StringIteratorLogic::Clone( void ) const
{
	return *new StringIteratorLogic(*this);
}

asli::IteratorLogic& asli::StringIteratorLogic::operator++( void )
{
	++_it;
	return *this;
}

asli::IteratorLogic& asli::StringIteratorLogic::operator++( int )
{
	StringIteratorLogic *copy = new StringIteratorLogic(_it);
	++_it;
	return *copy;
}

const axis::String::value_type& asli::StringIteratorLogic::operator*( void ) const
{
	return *_it;
}

bool asli::StringIteratorLogic::operator==( const IteratorLogic& it ) const
{
	const StringIteratorLogic& other = (const StringIteratorLogic&)it;
	return this->_it == other._it;
}

bool asli::StringIteratorLogic::operator!=( const IteratorLogic& it ) const
{
	return !(*this == it);
}

bool asli::StringIteratorLogic::operator>( const IteratorLogic& other ) const
{
	const StringIteratorLogic& it = (const StringIteratorLogic&)other;
	return this->_it > it._it;
}