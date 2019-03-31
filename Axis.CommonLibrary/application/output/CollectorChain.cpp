#include "CollectorChain.hpp"
#include "collectors/NodeSetCollector.hpp"
#include "collectors/ElementSetCollector.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

// explicit instantiation
template class aao::CollectorChain<ade::Node, aaoc::NodeSetCollector>;
template class aao::CollectorChain<ade::FiniteElement, aaoc::ElementSetCollector>;

template<class CollectedItem, class Collector>
aao::CollectorChain<CollectedItem, Collector>::CollectorChain( void )
{
  // nothing to do here
}

template<class CollectedItem, class Collector>
aao::CollectorChain<CollectedItem, Collector>::~CollectorChain( void )
{
  // destroy every collector created 
  collector_list::iterator end = collectors_.end();
  for (collector_list::iterator it = collectors_.begin(); it != end; ++it)
  {
    (**it).Destroy();
  }
  collectors_.clear();
}

template<class CollectedItem, class Collector>
void axis::application::output::CollectorChain<CollectedItem, Collector>::Destroy( void ) const
{
  delete this;
}

template<class CollectedItem, class Collector>
bool aao::CollectorChain<CollectedItem, Collector>::IsOfInterest( const asmm::ResultMessage& message ) const
{
  collector_list::const_iterator end = collectors_.end();
  for (collector_list::const_iterator it = collectors_.begin(); it != end; ++it)
  {
    const Collector& c = **it;
    if (c.IsOfInterest(message)) return true;
  }
  return false;
}

template<class CollectedItem, class Collector>
void aao::CollectorChain<CollectedItem, Collector>::Add( const Collector& collector )
{
  // add collector if it hasn't been done so before
  Collector *c = const_cast<Collector *>(&collector); // we want only the pointer, so it is alright
  collector_list::const_iterator end = collectors_.end();
  for (collector_list::const_iterator it = collectors_.begin(); it != end; ++it)
  {
    const Collector *ptr = &(**it);
    if (ptr == c)
    { // whoops, it has already been added!
      throw axis::foundation::ArgumentException(_T("Collector already exists in this collection."));
    }
  }
  collectors_.push_back(c);
}

template<class CollectedItem, class Collector>
void aao::CollectorChain<CollectedItem, Collector>::Remove( const Collector& collector )
{
  Collector *c = const_cast<Collector *>(&collector); // we want only the pointer, so it is alright
  collector_list::const_iterator end = collectors_.end();
  for (collector_list::const_iterator it = collectors_.begin(); it != end; ++it)
  {
    const Collector *ptr = &(**it);
    if (ptr == c)
    { 
      collectors_.erase(it);
    }
  }
  // whoops, collector not found!
  throw axis::foundation::ElementNotFoundException(_T("Collector is not registered in this group."));  
}

template<class CollectedItem, class Collector>
bool axis::application::output::CollectorChain<CollectedItem, Collector>::IsEmpty( void ) const
{
  return collectors_.empty();
}

template<class CollectedItem, class Collector>
axis::String axis::application::output::CollectorChain<CollectedItem, Collector>::GetTargetSetName( void ) const
{
  if (collectors_.empty()) throw axis::foundation::InvalidOperationException(_T("Impossible to determine."));
  collector_list::const_iterator it = collectors_.begin();
  return (**it).GetTargetSetName();
}

template<class CollectedItem, class Collector>
int axis::application::output::CollectorChain<CollectedItem, Collector>::GetCollectorCount( void ) const
{
  return (int)collectors_.size();
}

template<class CollectedItem, class Collector>
const Collector& axis::application::output::CollectorChain<CollectedItem, Collector>::operator[]( int index ) const
{
  return *collectors_[index];
}

template<class CollectedItem, class Collector>
void aao::CollectorChain<CollectedItem, Collector>::ChainCollect( const asmm::ResultMessage& message, 
                                                            aaor::ResultRecordset& recordset,
                                                            const CollectedItem& item )
{
  // a chain is said to be interested in a particular 
  // message if the head collectors is interested in 
  // it; that is, all collectors must agree to work in
  // the exact same messages
  collector_list::iterator it = collectors_.begin();
  Collector& headCollector = **it;
  if (!headCollector.IsOfInterest(message))
  { // skip message
    return;
  }

  // forward message
  collector_list::iterator end = collectors_.end();
  for (; it != end; ++it)
  {
    Collector& c = **it;
    c.Collect(item, message, recordset);
  }
}
