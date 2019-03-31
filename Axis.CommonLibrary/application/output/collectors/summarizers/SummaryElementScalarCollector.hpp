#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "application/output/collectors/GenericCollector.hpp"
#include "SummaryType.hpp"

namespace axis { 

namespace domain { namespace elements {
class FiniteElement;
} } // namespace axis::domain::elements

namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Represents a collector that summarizes scalar data from elements.
 *
 * @sa GenericCollector
**/
class AXISCOMMONLIBRARY_API SummaryElementScalarCollector : 
  public GenericCollector
{
public:
  SummaryElementScalarCollector(const axis::String& targetSetName, 
    SummaryType summaryType);
  virtual ~SummaryElementScalarCollector(void);

  virtual void Collect(const axis::services::messaging::ResultMessage& message, 
    axis::application::output::recordsets::ResultRecordset& recordset, 
    const axis::domain::analyses::NumericalModel& numericalModel);
  virtual bool IsOfInterest( 
    const axis::services::messaging::ResultMessage& message ) const;
  virtual axis::String GetFriendlyDescription( void ) const;
private:
  virtual axis::String GetVariableName(void) const = 0;
  virtual void StartCollect(void);
  virtual real CollectValue(
    const axis::services::messaging::ResultMessage& message, 
    const axis::domain::elements::FiniteElement& element,
    const axis::domain::analyses::NumericalModel& numericalModel) = 0;

  axis::String targetSetName_;
  SummaryType summaryType_;
  real value_;
};

} } } } } // namespace axis::application::output::collectors::summarizers
