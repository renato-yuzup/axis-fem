#include "ElementEffectivePlasticStrainCollector.hpp"
#include <assert.h>
#include "domain/analyses/ModelKinematics.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "application/output/recordsets/ResultRecordset.hpp"

namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace asmm = axis::services::messaging;
namespace aaor = axis::application::output::recordsets;
namespace adam = axis::domain::algorithms::messages;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::ElementEffectivePlasticStrainCollector::
  ElementEffectivePlasticStrainCollector( const axis::String& targetSetName, 
  const axis::String& customFieldName ): ElementSetCollector(targetSetName), 
  fieldName_(customFieldName)
{
  // nothing to do here
}

aaoc::ElementEffectivePlasticStrainCollector& 
  aaoc::ElementEffectivePlasticStrainCollector::Create( 
  const axis::String& targetSetName )
{
  return Create(targetSetName, _T("Effective Plastic Strain"));
}

aaoc::ElementEffectivePlasticStrainCollector& 
  aaoc::ElementEffectivePlasticStrainCollector::Create( 
  const axis::String& targetSetName, const axis::String& customFieldName)
{
  return *new aaoc::ElementEffectivePlasticStrainCollector(targetSetName, 
    customFieldName);
}

aaoc::ElementEffectivePlasticStrainCollector::
  ~ElementEffectivePlasticStrainCollector( void )
{
  // nothing to do here
}

void aaoc::ElementEffectivePlasticStrainCollector::Destroy( void ) const
{
  delete this;
}

void aaoc::ElementEffectivePlasticStrainCollector::Collect( 
  const ade::FiniteElement& element, const asmm::ResultMessage&,                                              
  aaor::ResultRecordset& recordset)
{
  real effPlasticStrain = element.PhysicalState().EffectivePlasticStrain();
  recordset.WriteData(effPlasticStrain);
}

axis::String 
  aaoc::ElementEffectivePlasticStrainCollector::GetFieldName(void) const
{
  return fieldName_;
}

aao::DataType 
  aaoc::ElementEffectivePlasticStrainCollector::GetFieldType( void ) const
{
  return kRealNumber;
}

int aaoc::ElementEffectivePlasticStrainCollector::GetVectorFieldLength(void) const
{
  return 0;
}

axis::String 
  aaoc::ElementEffectivePlasticStrainCollector::GetFriendlyDescription(void) const
{
  axis::String description = _T("Effective plastic strain");
  if (GetTargetSetName().empty())
  {
    description += _T(" on all elements");
  }
  else
  {
    description += _T(" on element set '") + GetTargetSetName() + _T("'");
  }
  return description;
}