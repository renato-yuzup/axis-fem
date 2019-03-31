#pragma once
#include "TextReportNode6DCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

  class TextReportNodeStrainCollector : public TextReportNode6DCollector
  {
  public:
    TextReportNodeStrainCollector(const axis::String& targetSetName, const bool *activeDirections);
    ~TextReportNodeStrainCollector(void);
    virtual void Destroy( void ) const;
  private:
    virtual real GetDofData( const axis::domain::analyses::NumericalModel& numericalModel, 
      const axis::domain::elements::Node& node, int dofId );
    virtual axis::String GetVariableName( bool plural ) const;
    virtual axis::String GetVariableSymbol( void ) const;
  };

} } } } // namespace axis::application::output::collectors
