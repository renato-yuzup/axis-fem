#include "ElementSetCollector.hpp"
#include "foundation/ArgumentException.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"

namespace aaoc = axis::application::output::collectors;
namespace adam = axis::domain::algorithms::messages;
namespace asmm = axis::services::messaging;

aaoc::ElementSetCollector::ElementSetCollector( const axis::String& targetSetName ) :
targetSetName_(targetSetName)
{
  // nothing to do here
}

aaoc::ElementSetCollector::~ElementSetCollector( void )
{
  // nothing to do here
}

axis::String aaoc::ElementSetCollector::GetTargetSetName( void ) const
{
  return targetSetName_;
}

int axis::application::output::collectors::ElementSetCollector::GetMatrixFieldRowCount( void ) const
{
  return 0;
}

int axis::application::output::collectors::ElementSetCollector::GetMatrixFieldColumnCount( void ) const
{
  return 0;
}

int axis::application::output::collectors::ElementSetCollector::GetVectorFieldLength( void ) const
{
  return 0;
}

bool aaoc::ElementSetCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}
