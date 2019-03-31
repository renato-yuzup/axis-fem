#include "SnapshotEndMessage.hpp"
#include "domain/analyses/AnalysisInfo.hpp"

namespace adam = axis::domain::algorithms::messages;
namespace ada = axis::domain::analyses;
namespace asmm = axis::services::messaging;

const asmm::Message::id_type adam::SnapshotEndMessage::BaseId = 450;

adam::SnapshotEndMessage::SnapshotEndMessage( const ada::AnalysisInfo& analysisInfo ) :
	ResultMessage(BaseId), analysisInfo_(analysisInfo.Clone())
{
	// nothing to do here
}

adam::SnapshotEndMessage::SnapshotEndMessage( const ada::AnalysisInfo& analysisInfo, 
                                               const axis::String& description ) :
	ResultMessage(BaseId, description), analysisInfo_(analysisInfo.Clone())
{
	// nothing to do here	
}

adam::SnapshotEndMessage::~SnapshotEndMessage( void )
{
	analysisInfo_.Destroy();
}

const ada::AnalysisInfo& adam::SnapshotEndMessage::GetAnalysisInformation( void ) const
{
  return analysisInfo_;
}

bool adam::SnapshotEndMessage::IsOfKind( const ResultMessage& message )
{
	return message.GetId() == BaseId;
}

void adam::SnapshotEndMessage::DoDestroy( void ) const
{
	delete this;
}

asmm::Message& adam::SnapshotEndMessage::DoClone( id_type id ) const
{
	return *new SnapshotEndMessage(analysisInfo_, this->GetDescription());
}

