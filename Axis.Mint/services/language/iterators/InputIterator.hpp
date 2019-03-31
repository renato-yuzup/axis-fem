#pragma once
#include <iterator>
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"
#include "IteratorLogic.hpp"

namespace axis { namespace services { namespace language { namespace iterators {

class AXISMINT_API InputIterator
{
public:
  InputIterator(void);	// creates a "no-go" iterator
  InputIterator(const InputIterator& other);
  InputIterator(IteratorLogic& logicImpl);

  ~InputIterator(void);

  virtual InputIterator& Clone(void) const;

  InputIterator& operator ++(void);	// pre-fixed
  InputIterator operator ++(int);		// post-fixed

  bool operator >(const InputIterator& other) const;
  bool operator >=(const InputIterator& other) const;
  bool operator <(const InputIterator& other) const;
  bool operator <=(const InputIterator& other) const;

  const axis::String::value_type& operator *(void) const;

  axis::String ToString(const InputIterator& end) const;

  InputIterator& operator =(const InputIterator& it);

  bool operator ==(const InputIterator& it) const;
  bool operator !=(const InputIterator& it) const;
private:
  IteratorLogic *_iteratorImpl;
};			

} } } } // namespace axis::services::language::iterators

template<> struct std::iterator_traits<class axis::services::language::iterators::InputIterator>
{	// get traits from iterator _Iter
  typedef std::forward_iterator_tag iterator_category;
  typedef const axis::String::char_type value_type;
  typedef ptrdiff_t  difference_type;
  typedef value_type * pointer;
  typedef value_type& reference;
};


