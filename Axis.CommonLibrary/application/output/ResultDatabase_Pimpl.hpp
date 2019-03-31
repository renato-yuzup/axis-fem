#pragma once
#include "ResultDatabase.hpp"
#include <list>
#include <vector>
#include "CollectorGroup.hpp"
#include "collectors/NodeSetCollector.hpp"
#include "collectors/ElementSetCollector.hpp"
#include "collectors/GenericCollector.hpp"
#include "foundation/collections/BimapSet.hpp"

namespace ade = axis::domain::elements;
namespace aaoc = axis::application::output::collectors;

namespace axis { 

namespace domain { namespace elements {
  class Node;
  class FiniteElement;
} } // namespace axis::domain::elements

namespace application { namespace output {

namespace workbooks {
  class ResultWorkbook;
} // namespace axis::application::output::workbooks

/**
 * Pimpl of ResultDatabase.
 */
class ResultDatabase::Pimpl
{
public:
  typedef CollectorGroup<ade::Node, aaoc::NodeSetCollector> NodeSetCollectorGroup;
  typedef CollectorGroup<ade::FiniteElement, aaoc::ElementSetCollector> ElementSetCollectorGroup;
  typedef std::list<aaoc::GenericCollector *> GenericCollectorSet;
  typedef std::vector<aaoc::EntityCollector *> CollectorSet;

  ElementSetCollectorGroup ElementCollectors;
  NodeSetCollectorGroup NodeCollectors;
  GenericCollectorSet GenericCollectors;
  CollectorSet AllCollectors;
  bool IsOpen;

  workbooks::ResultWorkbook *CurrentWorkbook;

  Pimpl(void);
  ~Pimpl(void);
  void PopulateAndInitWorkbook(axis::application::jobs::WorkFolder& workFolder);
  void WriteNodalResults(const axis::services::messaging::ResultMessage& message,
    const axis::domain::analyses::NumericalModel& numericalModel);
  void WriteElementResults(const axis::services::messaging::ResultMessage& message,
    const axis::domain::analyses::NumericalModel& numericalModel);
  void WriteGenericResults(const axis::services::messaging::ResultMessage& message,
    const axis::domain::analyses::NumericalModel& numericalModel);
  void CreateRecordsetFields(void);
  void SetUpAllCollectors(void);
  void TearDownAllCollectors( void );
private:
  void CreateNodeRecordsetFields(void);
  void CreateElementRecordsetFields(void);
}; // ResultDatabase::Pimpl

} } } // namespace axis::application::output
