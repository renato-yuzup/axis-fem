#include "SummaryElementArtificialEnergyCollector.hpp"
#include "domain/elements/FiniteElement.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryElementArtificialEnergyCollector::SummaryElementArtificialEnergyCollector( 
  const axis::String& targetSetName, SummaryType summaryType ) :
SummaryElementScalarCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryElementArtificialEnergyCollector::~SummaryElementArtificialEnergyCollector( void )
{
  // nothing to do here
}

aaocs::SummaryElementArtificialEnergyCollector& aaocs::SummaryElementArtificialEnergyCollector::Create( 
    const axis::String& targetSetName, SummaryType summaryType )
{
  return *new SummaryElementArtificialEnergyCollector(targetSetName, summaryType);
}

void aaocs::SummaryElementArtificialEnergyCollector::Destroy( void ) const
{
  delete this;
}

axis::String aaocs::SummaryElementArtificialEnergyCollector::GetVariableName( void ) const
{
  return _T("artificial energy");
}

real aaocs::SummaryElementArtificialEnergyCollector::CollectValue( const asmm::ResultMessage&, 
                                                                   const ade::FiniteElement& element, 
                                                                   const ada::NumericalModel& )
{
  return element.GetTotalArtificialEnergy();
}
