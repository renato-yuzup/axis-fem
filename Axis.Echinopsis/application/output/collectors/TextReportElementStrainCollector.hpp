#pragma once
#include "TextReportElement6DCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

  class TextReportElementStrainCollector : public TextReportElement6DCollector
  {
  public:
    TextReportElementStrainCollector(const axis::String& targetSetName, const bool *activeDirections);
    ~TextReportElementStrainCollector(void);
    virtual void Destroy( void ) const;
  private:
    virtual real GetDofData( const axis::domain::analyses::NumericalModel& numericalModel, 
      const axis::domain::elements::FiniteElement& element, int directionIdx );
    virtual axis::String GetVariableName( bool plural ) const;
    virtual axis::String GetVariableSymbol( void ) const;
  };

} } } } // namespace axis::application::output::collectors
