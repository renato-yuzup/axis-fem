#include "ElementDeformationGradientCollector.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "application/output/recordsets/ResultRecordset.hpp"

namespace aao  = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ade  = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb  = axis::foundation::blas;

aaoc::ElementDeformationGradientCollector::ElementDeformationGradientCollector( 
  const axis::String& targetSetName, const axis::String& customFieldName ) : 
  ElementSetCollector(targetSetName), fieldName_(customFieldName)
{
  // nothing to do here
}

aaoc::ElementDeformationGradientCollector& 
  aaoc::ElementDeformationGradientCollector::Create( 
  const axis::String& targetSetName )
{
  return *new aaoc::ElementDeformationGradientCollector(targetSetName, 
    _T("Deformation gradient"));
}

aaoc::ElementDeformationGradientCollector& 
  aaoc::ElementDeformationGradientCollector::Create( 
  const axis::String& targetSetName, const axis::String& customFieldName )
{
  return *new aaoc::ElementDeformationGradientCollector(targetSetName, 
    customFieldName);
}

aaoc::ElementDeformationGradientCollector::~ElementDeformationGradientCollector( 
  void )
{
  // nothing to do here
}

axis::String aaoc::ElementDeformationGradientCollector::GetFieldName( 
  void ) const
{
  return fieldName_;
}

aao::DataType aaoc::ElementDeformationGradientCollector::GetFieldType( 
  void ) const
{
  return aao::kMatrix;
}

int aaoc::ElementDeformationGradientCollector::GetMatrixFieldRowCount( 
  void ) const
{
  return 3;
}

int aaoc::ElementDeformationGradientCollector::GetMatrixFieldColumnCount( 
  void ) const
{
  return 3;
}

void aaoc::ElementDeformationGradientCollector::Collect( 
  const ade::FiniteElement& element, const asmm::ResultMessage& message, 
  aaor::ResultRecordset& recordset )
{
  const auto& eState = element.PhysicalState();
  const auto& F = eState.DeformationGradient();
  recordset.WriteData(F);
}

void aaoc::ElementDeformationGradientCollector::Destroy( void ) const
{
  delete this;
}

axis::String aaoc::ElementDeformationGradientCollector::GetFriendlyDescription( 
  void ) const
{
  axis::String description = _T("Deformation gradient tensor");
  if (GetTargetSetName().empty())
  {
    description += _T(" of all elements");
  }
  else
  {
    description += _T(" of element set '") + GetTargetSetName() + _T("'");
  }
  return description;
}
