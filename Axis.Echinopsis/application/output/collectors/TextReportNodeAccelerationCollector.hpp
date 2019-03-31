#pragma once
#include "TextReportNode3DCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

  class TextReportNodeAccelerationCollector : public TextReportNode3DCollector
  {
  public:
    TextReportNodeAccelerationCollector(const axis::String& targetSetName, const bool *activeDirections);
    ~TextReportNodeAccelerationCollector(void);
    virtual void Destroy( void ) const;
  private:
    virtual real GetDofData( const axis::domain::analyses::NumericalModel& numericalModel, 
      const axis::domain::elements::Node& node, int dofId );
    virtual axis::String GetVariableName( bool plural ) const;
    virtual axis::String GetVariableSymbol( void ) const;
  };

} } } } // namespace axis::application::output::collectors
