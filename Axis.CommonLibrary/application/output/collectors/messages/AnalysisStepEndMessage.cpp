#include "AnalysisStepEndMessage.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace aaj = axis::application::jobs;
namespace asmm = axis::services::messaging;

const id_type aaocm::AnalysisStepEndMessage::BaseId = 350;

aaocm::AnalysisStepEndMessage::AnalysisStepEndMessage( void ) : ResultMessage(BaseId)
{
  // nothing to do here
}

aaocm::AnalysisStepEndMessage::~AnalysisStepEndMessage( void )
{
	// nothing to do here
}

void aaocm::AnalysisStepEndMessage::DoDestroy( void ) const
{
	delete this;
}

asmm::Message& aaocm::AnalysisStepEndMessage::DoClone( id_type ) const
{
	return *new AnalysisStepEndMessage();
}

bool aaocm::AnalysisStepEndMessage::IsOfKind( const asmm::Message& message )
{
	return BaseId == message.GetId();
}
