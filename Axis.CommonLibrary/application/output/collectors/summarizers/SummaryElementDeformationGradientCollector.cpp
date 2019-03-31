#include "SummaryElementDeformationGradientCollector.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaocs::SummaryElementDeformationGradientCollector::
  SummaryElementDeformationGradientCollector(const axis::String& targetSetName, 
  SummaryType summaryType) : SummaryElementMatrixCollector(targetSetName, 
  summaryType, 3, 3)
{
  // nothing to do here
}
  
aaocs::SummaryElementDeformationGradientCollector& 
  aaocs::SummaryElementDeformationGradientCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new SummaryElementDeformationGradientCollector(targetSetName, 
    summaryType);
}

aaocs::SummaryElementDeformationGradientCollector::
  ~SummaryElementDeformationGradientCollector(void)
{
  // nothing to do here
}

void aaocs::SummaryElementDeformationGradientCollector::Destroy( void ) const
{
  delete this;
}

axis::String aaocs::SummaryElementDeformationGradientCollector::GetVariableName( 
  void ) const
{
  return _T("deformation gradient");
}

real aaocs::SummaryElementDeformationGradientCollector::CalculateMatrixNorm( 
  const asmm::ResultMessage& message, const ade::FiniteElement& element )
{
  auto& F = element.PhysicalState().DeformationGradient();
  real f11, f12, f13, f21, f22, f23, f31, f32, f33;
  f11 = F(0,0);    f12 = F(0,1);    f13 = F(0,2);
  f21 = F(1,0);    f22 = F(1,1);    f23 = F(1,2);
  f31 = F(2,0);    f32 = F(2,1);    f33 = F(2,2);
  afb::SymmetricMatrix C(3);
  afb::Product(C, 1.0, F, afb::NotTransposed, F, afb::Transposed);
  real equivalent = afb::DoubleContraction(1.0, C, 1.0, C);
  equivalent = sqrt(2.0 / 3.0 * equivalent);
  return equivalent;
}

const afb::DenseMatrix& 
  aaocs::SummaryElementDeformationGradientCollector::CollectMatrix( 
  const asmm::ResultMessage&, const ade::FiniteElement& element )
{
  return element.PhysicalState().DeformationGradient();
}
