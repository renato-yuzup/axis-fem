#include "CollectorGroup.hpp"
#include "collectors/NodeSetCollector.hpp"
#include "collectors/ElementSetCollector.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/ArgumentException.hpp"

namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace ade = axis::domain::elements;
namespace afc = axis::foundation::collections;

// explicit instantiation
#include "foundation/collections/MappedSet.cpp"
template class aao::CollectorGroup<ade::Node, aaoc::NodeSetCollector>;
template class aao::CollectorGroup<ade::FiniteElement, aaoc::ElementSetCollector>;

template <class CollectedItem, class Collector>
aao::CollectorGroup<CollectedItem, Collector>::CollectorGroup(void)
{
  // nothing to do here
}

template <class CollectedItem, class Collector>
aao::CollectorGroup<CollectedItem, Collector>::~CollectorGroup(void)
{
  // destroy all chains
  groups_.DestroyChildren();
}

template<class CollectedItem, class Collector>
void aao::CollectorGroup<CollectedItem, Collector>::AddCollector( const Collector& collector )
{
  // lookup for the set collector list
  axis::String setName = collector.GetTargetSetName();
  Chain *collectorList;
  if (!groups_.Contains(setName))
  { // not found, create one
    collectorList = new Chain();
    groups_.Add(setName, *collectorList);
  }
  else
  {
    collectorList = &groups_[setName];
  }
  collectorList->Add(collector);  
}

template<class CollectedItem, class Collector>
void aao::CollectorGroup<CollectedItem, Collector>::RemoveCollector( const Collector& collector )
{
  // lookup for the set collector list
  axis::String setName = collector.GetTargetSetName();
  Chain *collectorList;
  if (!groups_.Contains(setName))
  { // whoops, target set not found!
    throw axis::foundation::ElementNotFoundException(_T("Collector is not registered in this group."));
  }
  else
  {
    collectorList = &groups_[setName];
  }
  collectorList->Remove(collector);
  if (collectorList->IsEmpty())
  {
    delete collectorList;
    groups_.Remove(setName);
  }
}

template<class CollectedItem, class Collector>
typename aao::CollectorGroup<CollectedItem, Collector>::Chain& 
    aao::CollectorGroup<CollectedItem, Collector>::operator[]( const axis::String& targetSetName ) const
{
  return groups_[targetSetName];
}

template<class CollectedItem, class Collector>
typename aao::CollectorGroup<CollectedItem, Collector>::Chain& 
    aao::CollectorGroup<CollectedItem, Collector>::operator[]( int index ) const
{
  return groups_[index];
}

template<class CollectedItem, class Collector>
int aao::CollectorGroup<CollectedItem, Collector>::GetChainCount( void ) const
{
  return groups_.Count();
}
