#include "NodeSetCollector.hpp"
#include "foundation/ArgumentException.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"

namespace aaoc = axis::application::output::collectors;
namespace adam = axis::domain::algorithms::messages;
namespace asmm = axis::services::messaging;

aaoc::NodeSetCollector::NodeSetCollector( const axis::String& targetSetName ) :
targetSetName_(targetSetName)
{
  // nothing to do here
}

aaoc::NodeSetCollector::~NodeSetCollector( void )
{
  // nothing to do here
}

axis::String aaoc::NodeSetCollector::GetTargetSetName( void ) const
{
  return targetSetName_;
}

int aaoc::NodeSetCollector::GetMatrixFieldRowCount( void ) const
{
  return 0;
}

int aaoc::NodeSetCollector::GetMatrixFieldColumnCount( void ) const
{
  return 0;
}

int aaoc::NodeSetCollector::GetVectorFieldLength( void ) const
{
  return 0;
}

bool aaoc::NodeSetCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}
