#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "application/output/collectors/GenericCollector.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "SummaryType.hpp"

namespace axis { 

namespace domain { namespace elements {
class Node;
} } // namespace axis::domain::elements

namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Represents a collector that summarizes three-dimensional data from nodes.
 *
 * @sa GenericCollector
**/
class AXISCOMMONLIBRARY_API SummaryNode3DCollector : public GenericCollector
{
public:
  SummaryNode3DCollector(const axis::String& targetSetName, SummaryType summaryType);
  SummaryNode3DCollector(const axis::String& targetSetName, SummaryType summaryType,
                         axis::application::output::collectors::XDirectionState xState,
                         axis::application::output::collectors::YDirectionState yState,
                         axis::application::output::collectors::ZDirectionState zState);
  virtual ~SummaryNode3DCollector(void);

  virtual void Collect(const axis::services::messaging::ResultMessage& message, 
                       axis::application::output::recordsets::ResultRecordset& recordset, 
                       const axis::domain::analyses::NumericalModel& numericalModel);
  virtual bool IsOfInterest( const axis::services::messaging::ResultMessage& message ) const;
  virtual axis::String GetFriendlyDescription( void ) const;
private:
  virtual axis::String GetVariableName(bool plural) const = 0;
  virtual void StartCollect(void);
  virtual real CollectValue(const axis::services::messaging::ResultMessage& message, 
                            const axis::domain::elements::Node& node, int directionIndex,
                            const axis::domain::analyses::NumericalModel& numericalModel) = 0;
  virtual void Summarize(axis::application::output::recordsets::ResultRecordset& recordset);
  
  axis::String targetSetName_;
  SummaryType summaryType_;
  real values_[3];
  bool state_[3];
  int vectorSize_;
};

} } } } } // namespace axis::application::output::collectors::summarizers
