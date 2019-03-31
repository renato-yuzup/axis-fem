#include "IteratorFactory.hpp"
#include "../iterators/StringIteratorLogic.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;

aslf::IteratorFactory::IteratorFactory( void )
{
	/* 
		Default implementation -- this constructor is declared as private in
		order to make this class not instantiable.
	*/
}

asli::InputIterator aslf::IteratorFactory::CreateStringIterator( const axis::String& inputString )
{
	return asli::InputIterator(*new asli::StringIteratorLogic(inputString));
}

asli::InputIterator aslf::IteratorFactory::CreateStringIterator( 
  const axis::String::const_iterator& sourceIterator )
{
	return asli::InputIterator(*new asli::StringIteratorLogic(sourceIterator));
}