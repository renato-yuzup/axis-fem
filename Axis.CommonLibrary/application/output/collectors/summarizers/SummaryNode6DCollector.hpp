#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "application/output/collectors/GenericCollector.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "SummaryType.hpp"

namespace axis { 

namespace domain { namespace elements {
class Node;
} } // namespace axis::domain::elements

namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Represents a collector that summarizes 3D-tensor data from nodes.
 *
 * @sa GenericCollector
**/
class AXISCOMMONLIBRARY_API SummaryNode6DCollector : public GenericCollector
{
public:
  SummaryNode6DCollector(const axis::String& targetSetName, SummaryType summaryType);
  SummaryNode6DCollector(const axis::String& targetSetName, SummaryType summaryType,
                          axis::application::output::collectors::XXDirectionState xxState,
                          axis::application::output::collectors::YYDirectionState yyState,
                          axis::application::output::collectors::ZZDirectionState zzState,
                          axis::application::output::collectors::YZDirectionState yzState,
                          axis::application::output::collectors::XZDirectionState xzState,
                          axis::application::output::collectors::XYDirectionState xyState);
  virtual ~SummaryNode6DCollector(void);

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
  real values_[6];
  bool state_[6];
  int vectorSize_;
};

} } } } } // namespace axis::application::output::collectors::summarizers
