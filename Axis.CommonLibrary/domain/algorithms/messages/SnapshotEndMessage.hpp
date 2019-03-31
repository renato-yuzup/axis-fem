#pragma once
#include "services/messaging/ResultMessage.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { 

namespace analyses {
class AnalysisInfo;
} // namespace axis::domain::analyses

namespace algorithms { namespace messages {
  
/**
 * Message dispatched when a data collection round ends.
 *
 * @sa axis::services::messaging::ResultMessage
 */
class AXISCOMMONLIBRARY_API SnapshotEndMessage : public axis::services::messaging::ResultMessage
{
public:
	SnapshotEndMessage(const axis::domain::analyses::AnalysisInfo& analysisInfo);
	SnapshotEndMessage(const axis::domain::analyses::AnalysisInfo& analysisInfo, const axis::String& description);
	virtual ~SnapshotEndMessage(void);

  const axis::domain::analyses::AnalysisInfo& GetAnalysisInformation(void) const;

	static bool IsOfKind(const ResultMessage& message);

	virtual void DoDestroy( void ) const;

	virtual Message& DoClone( id_type id ) const;

	static const Message::id_type BaseId;
private:
  const axis::domain::analyses::AnalysisInfo& analysisInfo_;
}; // SnapshotEndMessage

} } } } // namespace axis::domain::algorithms::messages

