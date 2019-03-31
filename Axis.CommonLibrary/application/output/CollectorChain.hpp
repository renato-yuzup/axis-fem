#pragma once
#include <vector>
#include "AxisString.hpp"
#include "foundation/collections/Collectible.hpp"

namespace axis { 

namespace services { namespace messaging {
  class ResultMessage;
} } // namespace axis::services::messaging

namespace application { namespace output {

namespace recordsets {
  class ResultRecordset;
} // namespace axis::application::output::recordsets::ResultRecordset

/**
 * Groups entity collectors according to which 
 * entity set they work on.
 */
template<class CollectedItem, class Collector>
class CollectorChain : public axis::foundation::collections::Collectible
{
public:
  CollectorChain(void);
  ~CollectorChain(void);

  /**
   * Destroys this object.
   */
  void Destroy(void) const;

  /**
   * Queries if 'message' is of interest for this chain.
   *
   * @param message The message.
   *
   * @return true if it is of interest, false otherwise.
   */
  bool IsOfInterest(const axis::services::messaging::ResultMessage& message) const;

  /**
   * Adds a new collector to the end of the chain.
   *
   * @param collector The collector to add.
   * @remarks Although not checked, the new collector must accept the same set of messages
   *          than other collectors already in the chain, in order to avoid undefined
   *          behavior.
   */
  void Add(const Collector& collector);

  /**
   * Removes the given collector from the chain.
   *
   * @param collector The collector to remove.
   */
  void Remove(const Collector& collector);

  /**
   * Queries if this chain has no collector.
   *
   * @return true if it is empty, false otherwise.
   */
  bool IsEmpty(void) const;

  /**
   * Returns the name of the entity set this chain is targeting to.
   *
   * @return The target set name.
   */
  axis::String GetTargetSetName(void) const;

  /**
   * Returns how many collector are in this chain.
   *
   * @return The collector count.
   */
  int GetCollectorCount(void) const;

  /**
   * Returns a collector in the chain.
   *
   * @param index Zero-based index of the collector.
   *
   * @return The collector.
   */
  const Collector& operator[](int index) const;

  /**
   * Requests to process a message by every collector in this chain.
   *
   * @param message            The message to process.
   * @param [in,out] recordset The recordset to which data should be written to.
   * @param item               The item on which data will be based.
   */
  void ChainCollect(const axis::services::messaging::ResultMessage& message, 
                    axis::application::output::recordsets::ResultRecordset& recordset,
                    const CollectedItem& item);
private:
  typedef std::vector<Collector *> collector_list;
  collector_list collectors_;
}; // CollectorChain<CollectedItem, Collector>

} } } // namespace axis::application::output
