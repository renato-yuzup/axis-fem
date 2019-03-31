#include "SnapshotStartMessage.hpp"
#include "domain/analyses/AnalysisInfo.hpp"

namespace adam = axis::domain::algorithms::messages;
namespace ada = axis::domain::analyses;
namespace asmm = axis::services::messaging;

const axis::services::messaging::Message::id_type adam::SnapshotStartMessage::BaseId = 400;

adam::SnapshotStartMessage::SnapshotStartMessage( const ada::AnalysisInfo& analysisInfo ) :
ResultMessage(BaseId), analysisInfo_(analysisInfo.Clone())
{
	// nothing to do here
}

adam::SnapshotStartMessage::SnapshotStartMessage( const ada::AnalysisInfo& analysisInfo, 
                                                   const axis::String& description ) :
ResultMessage(BaseId, description), analysisInfo_(analysisInfo.Clone())
{
	// nothing to do here	
}

adam::SnapshotStartMessage::~SnapshotStartMessage( void )
{
	analysisInfo_.Destroy();
}

const ada::AnalysisInfo& adam::SnapshotStartMessage::GetAnalysisInformation( void ) const
{
  return analysisInfo_;
}

bool adam::SnapshotStartMessage::IsOfKind( const ResultMessage& message )
{
	return message.GetId() == BaseId;
}

void adam::SnapshotStartMessage::DoDestroy( void ) const
{
	delete this;
}

axis::services::messaging::Message& adam::SnapshotStartMessage::DoClone( id_type id ) const
{
	return *new SnapshotStartMessage(analysisInfo_, this->GetDescription());
}
