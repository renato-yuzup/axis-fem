#pragma once
#include "AxisString.hpp"
#include "CollectorChain.hpp"
#include "foundation/collections/MappedSet.hpp"

namespace axis { 
  
namespace domain { namespace elements {
class Node;
class FiniteElement;
} } // namespace axis::domain::elements

namespace application { namespace output {

namespace collectors {
class NodeSetCollector;
class ElementSetCollector;
} // namespace axis::application::output::collectors

/**
 * Groups entity collectors according to which 
 * entity set they work on.
 */
template<class CollectedItem, class Collector>
class CollectorGroup
{
public:
  typedef typename CollectorChain<CollectedItem, Collector> Chain;

  CollectorGroup(void);
  ~CollectorGroup(void);

  /**
   * Adds a collector to capture results.
   *
   * @param collector The collector.
   * @remarks If the collector operates on entity sets, it will be added to an existing collector
   *          chain for the referred entity set, if exists, or a new one will be created. It should
   *          be noted, however, that the newly added collector should accept the same set of messages
   *          than other collectors in the same chain where it will be added.
   */
  void AddCollector(const Collector& collector);

  /**
   * Removes a collector.
   *
   * @param collector The collector to remove.
   * @remarks If the collector is the last one in a chain, the chain is also destroyed.
   */
  void RemoveCollector(const Collector& collector);

  /**
   * Returns a collector chain in this group.
   *
   * @param targetSetName Name of the entity set that the chain is targeting.
   *
   * @return The collector chain.
   */
  Chain& operator [](const axis::String& targetSetName) const;

  /**
   * Returns a collector chain in this group.
   *
   * @param index Zero-based index of the collector chain.
   *
   * @return The collector chain.
   */
  Chain& operator [](int index) const;

  /**
   * Returns how many collector chains there are in this group.
   *
   * @return The chain count.
   */
  int GetChainCount(void) const;
private:
  typedef axis::foundation::collections::MappedSet<axis::String, Chain> GroupList;
  GroupList groups_;
}; // CollectorGroup<CollectedItem, Collector>

typedef axis::application::output::CollectorGroup<
          axis::domain::elements::Node, 
          axis::application::output::collectors::NodeSetCollector> NodeSetCollectorGroup;
typedef axis::application::output::CollectorGroup<
          axis::domain::elements::FiniteElement, 
          axis::application::output::collectors::ElementSetCollector> ElementSetCollectorGroup;

} } } // namespace axis::application::output
