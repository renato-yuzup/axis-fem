#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "SummaryElementMatrixCollector.hpp"

namespace axis { namespace application { namespace output { 
namespace collectors { namespace summarizers {

class AXISCOMMONLIBRARY_API SummaryElementDeformationGradientCollector :
  public SummaryElementMatrixCollector
{
public:
  static SummaryElementDeformationGradientCollector& Create(
    const axis::String& targetSetName, SummaryType summaryType);
  ~SummaryElementDeformationGradientCollector(void);
  virtual void Destroy( void ) const;
private:
  SummaryElementDeformationGradientCollector(const axis::String& targetSetName, 
    SummaryType summaryType);
  virtual axis::String GetVariableName( void ) const;
  virtual real CalculateMatrixNorm( 
    const axis::services::messaging::ResultMessage& message, 
    const axis::domain::elements::FiniteElement& element );
  virtual const axis::foundation::blas::DenseMatrix& CollectMatrix( 
    const axis::services::messaging::ResultMessage& message, 
    const axis::domain::elements::FiniteElement& element );
};

} } } } } // namespace axis::application::output::collectors::summarizers
