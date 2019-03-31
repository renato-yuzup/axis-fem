#include "ModelStateUpdateMessage.hpp"

namespace adam = axis::domain::algorithms::messages;
namespace asmm = axis::services::messaging;
namespace ada = axis::domain::analyses;

const int adam::ModelStateUpdateMessage::MessageId = 500;

adam::ModelStateUpdateMessage::ModelStateUpdateMessage( const ada::ModelKinematics& kinematicState,
                                                        const ada::ModelDynamics& dynamicState ) :
ResultMessage(MessageId), kinematicState_(kinematicState), dynamicState_(dynamicState)
{
  // nothing to do here
}

adam::ModelStateUpdateMessage::~ModelStateUpdateMessage( void )
{
  // nothing to do here
}

void adam::ModelStateUpdateMessage::DoDestroy( void ) const
{
  delete this;
}

asmm::Message& adam::ModelStateUpdateMessage::DoClone( id_type ) const
{
  return *new ModelStateUpdateMessage(kinematicState_, dynamicState_);
}

const ada::ModelKinematics& adam::ModelStateUpdateMessage::GetMeshKinematicState( void ) const
{
  return kinematicState_;
}

const ada::ModelDynamics& adam::ModelStateUpdateMessage::GetMeshDynamicState( void ) const
{
  return dynamicState_;
}

bool adam::ModelStateUpdateMessage::IsOfKind( const asmm::Message& message )
{
  return message.GetId() == MessageId;
}
