#pragma once
#include "services/messaging/ResultMessage.hpp"
#include "foundation/Axis.CommonLibrary.hpp"
#include "application/jobs/AnalysisStepInformation.hpp"

namespace axis { 

namespace foundation { namespace date_time {
class Timestamp;
} } // namespace axis::foundation::date_time
  
namespace domain { namespace analyses {
class AnalysisTimeline;
} } // namespace axis::domain::analyses

namespace application { 

namespace jobs {
class WorkFolder;
} // namespace axis::application::jobs

namespace output { namespace collectors { namespace messages {

/**
 * Message dispatched when an analysis step starts.
 *
 * @sa axis::services::messaging::ResultMessage
 */
class AXISCOMMONLIBRARY_API AnalysisStepStartMessage : public axis::services::messaging::ResultMessage
{
public:
	static const id_type BaseId;

	AnalysisStepStartMessage(const axis::application::jobs::AnalysisStepInformation& stepInfo);
	~AnalysisStepStartMessage(void);

  axis::application::jobs::AnalysisStepInformation GetStepInformation(void) const;

	virtual void DoDestroy( void ) const;

	virtual axis::services::messaging::Message& DoClone( id_type id ) const;


	static bool IsOfKind(const axis::services::messaging::Message& message);
private:
  axis::application::jobs::AnalysisStepInformation stepInfo_;
}; // AnalysisStepStartMessage

} } } } } // namespace axis::application::output::collectors::messages

