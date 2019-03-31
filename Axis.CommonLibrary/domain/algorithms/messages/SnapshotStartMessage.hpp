#pragma once
#include "services/messaging/ResultMessage.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { 
  
namespace analyses {
class AnalysisInfo;
} // namespace axis::domain::analyses
  
namespace algorithms { namespace messages {

/**
 * Message dispatched when a data collection round starts.
 *
 * @sa axis::services::messaging::ResultMessage
 */
class AXISCOMMONLIBRARY_API SnapshotStartMessage : public axis::services::messaging::ResultMessage
{
public:
	SnapshotStartMessage(const axis::domain::analyses::AnalysisInfo& analysisInfo);
	SnapshotStartMessage(const axis::domain::analyses::AnalysisInfo& analysisInfo, const axis::String& description);
	virtual ~SnapshotStartMessage(void);

  const axis::domain::analyses::AnalysisInfo& GetAnalysisInformation(void) const;

	static bool IsOfKind(const ResultMessage& message);

	virtual void DoDestroy( void ) const;

	virtual Message& DoClone( id_type id ) const;

	static const Message::id_type BaseId;
private:
  const axis::domain::analyses::AnalysisInfo& analysisInfo_;
}; // SnapshotStartMessage	

} } } } // namespace axis::domain::algorithms::messages

