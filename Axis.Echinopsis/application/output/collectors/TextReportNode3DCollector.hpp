#pragma once
#include "domain/collections/NodeSet.hpp"
#include "domain/elements/Node.hpp"
#include "services/messaging/ResultMessage.hpp"
#include "application/output/collectors/GenericCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

class TextReportNode3DCollector : public GenericCollector
{
private:
  bool dofsToPrint_[3];
  axis::String targetSetName_;
public:
  TextReportNode3DCollector(const axis::String& targetSetName, const bool *activeDirections);
  virtual ~TextReportNode3DCollector(void);

  virtual void Collect( const axis::services::messaging::ResultMessage& message, 
                        axis::application::output::recordsets::ResultRecordset& recordset, 
                        const axis::domain::analyses::NumericalModel& numericalModel );
  virtual bool IsOfInterest( const axis::services::messaging::ResultMessage& message ) const;
  virtual axis::String GetFriendlyDescription( void ) const;
private:
  virtual axis::String GetVariableName(bool plural) const = 0;
  virtual axis::String GetVariableSymbol(void) const = 0;
  virtual real GetDofData(const axis::domain::analyses::NumericalModel& numericalModel, 
                          const axis::domain::elements::Node& node, int dofId) = 0;
  void PrintHeader(axis::application::output::recordsets::ResultRecordset& recordset);
  axis::String BuildColumnHeader(void);
};

} } } }
