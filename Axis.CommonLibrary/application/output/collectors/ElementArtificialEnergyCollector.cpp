#include "ElementArtificialEnergyCollector.hpp"
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

aaoc::ElementArtificialEnergyCollector::ElementArtificialEnergyCollector( 
    const axis::String& targetSetName, const axis::String& customFieldName )
: ElementSetCollector(targetSetName), fieldName_(customFieldName)
{
  // nothing to do here
}

aaoc::ElementArtificialEnergyCollector& aaoc::ElementArtificialEnergyCollector::Create( const axis::String& targetSetName )
{
  return Create(targetSetName, _T("Artificial Energy"));
}

aaoc::ElementArtificialEnergyCollector& aaoc::ElementArtificialEnergyCollector::Create( 
  const axis::String& targetSetName, const axis::String& customFieldName)
{
  return *new aaoc::ElementArtificialEnergyCollector(targetSetName, customFieldName);
}

aaoc::ElementArtificialEnergyCollector::~ElementArtificialEnergyCollector( void )
{
  // nothing to do here
}

void aaoc::ElementArtificialEnergyCollector::Destroy( void ) const
{
  delete this;
}

void aaoc::ElementArtificialEnergyCollector::Collect( const ade::FiniteElement& element, 
                                                      const asmm::ResultMessage&,                                              
                                                      aaor::ResultRecordset& recordset)
{
  real energy = element.GetTotalArtificialEnergy();
  recordset.WriteData(energy);
}

axis::String aaoc::ElementArtificialEnergyCollector::GetFieldName( void ) const
{
  return fieldName_;
}

aao::DataType aaoc::ElementArtificialEnergyCollector::GetFieldType( void ) const
{
  return kRealNumber;
}

int aaoc::ElementArtificialEnergyCollector::GetVectorFieldLength( void ) const
{
  return 0;
}

axis::String aaoc::ElementArtificialEnergyCollector::GetFriendlyDescription( void ) const
{
  axis::String description = _T("Artificial energy");
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
