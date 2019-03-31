#pragma once
#include "services/messaging/ResultMessage.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { 

namespace jobs {
class AnalysisStep;
class WorkFolder;
} // namespace axis::application::jobs

namespace output { namespace collectors { namespace messages {

/**
 * Message dispatched when an analysis step ends.
 *
 * @sa axis::services::messaging::ResultMessage
 */
class AXISCOMMONLIBRARY_API AnalysisStepEndMessage : public axis::services::messaging::ResultMessage
{
public:
	static const id_type BaseId;

  AnalysisStepEndMessage(void);
	~AnalysisStepEndMessage(void);

	virtual void DoDestroy( void ) const;

	virtual axis::services::messaging::Message& DoClone( id_type id ) const;

  static bool IsOfKind(const axis::services::messaging::Message& message);
}; // AnalysisStepEndMessage

} } } } } // namespace axis::application::output::collectors::messages
