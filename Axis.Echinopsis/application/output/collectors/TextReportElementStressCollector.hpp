#pragma once
#include "TextReportElement6DCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

  class TextReportElementStressCollector : public TextReportElement6DCollector
  {
  public:
    TextReportElementStressCollector(const axis::String& targetSetName, const bool *activeDirections);
    ~TextReportElementStressCollector(void);
    virtual void Destroy( void ) const;
  private:
    virtual real GetDofData( const axis::domain::analyses::NumericalModel& numericalModel, 
                             const axis::domain::elements::FiniteElement& element, int directionIdx );
    virtual axis::String GetVariableName( bool plural ) const;
    virtual axis::String GetVariableSymbol( void ) const;
  };

} } } } // namespace axis::application::output::collectors
