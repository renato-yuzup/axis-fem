#include "AnalysisStepStartMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaj = axis::application::jobs;
namespace ada = axis::domain::analyses;
namespace asmm = axis::services::messaging;
namespace afd = axis::foundation::date_time;
namespace afu = axis::foundation::uuids;
namespace asdi = axis::services::diagnostics::information;

const asmm::Message::id_type aaocm::AnalysisStepStartMessage::BaseId = 300;


aaocm::AnalysisStepStartMessage::AnalysisStepStartMessage( const aaj::AnalysisStepInformation& stepInfo ) :
ResultMessage(BaseId), stepInfo_(stepInfo)
{
  // nothing to do here
}

aaocm::AnalysisStepStartMessage::~AnalysisStepStartMessage( void )
{
	// nothing to do here
}

aaj::AnalysisStepInformation aaocm::AnalysisStepStartMessage::GetStepInformation( void ) const
{
  return stepInfo_;
}

void aaocm::AnalysisStepStartMessage::DoDestroy( void ) const
{
	delete this;
}

asmm::Message& aaocm::AnalysisStepStartMessage::DoClone( id_type id ) const
{
	return *new AnalysisStepStartMessage(stepInfo_);
}

bool aaocm::AnalysisStepStartMessage::IsOfKind( const asmm::Message& message )
{
  return message.GetId() == BaseId;
}

