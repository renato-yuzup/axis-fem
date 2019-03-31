#pragma once
#include "domain/collections/ElementSet.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "services/messaging/ResultMessage.hpp"
#include "application/output/collectors/GenericCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

  class TextReportElement6DCollector : public GenericCollector
  {
  private:
    bool dofsToPrint_[6];
    axis::String targetSetName_;
  public:
    TextReportElement6DCollector(const axis::String& targetSetName, const bool *activeDirections);
    virtual ~TextReportElement6DCollector(void);

    virtual void Collect( const axis::services::messaging::ResultMessage& message, 
                          axis::application::output::recordsets::ResultRecordset& recordset, 
                          const axis::domain::analyses::NumericalModel& numericalModel );
    virtual bool IsOfInterest( const axis::services::messaging::ResultMessage& message ) const;
    virtual axis::String GetFriendlyDescription( void ) const;
  private:
    virtual axis::String GetVariableName(bool plural) const = 0;
    virtual axis::String GetVariableSymbol(void) const = 0;
    virtual real GetDofData(const axis::domain::analyses::NumericalModel& numericalModel, 
                            const axis::domain::elements::FiniteElement& element, int directionIdx) = 0;
    void PrintHeader(axis::application::output::recordsets::ResultRecordset& recordset);
    axis::String BuildColumnHeader(void);
  };

} } } }
