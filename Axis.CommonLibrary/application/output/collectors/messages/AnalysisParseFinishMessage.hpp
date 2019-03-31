#pragma once
#include "services/messaging/ResultMessage.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace messages {

class AXISCOMMONLIBRARY_API AnalysisParseFinishMessage : public axis::services::messaging::ResultMessage
{
public:
	AnalysisParseFinishMessage(void);
	AnalysisParseFinishMessage(const axis::String& description);
	virtual ~AnalysisParseFinishMessage(void);

	static bool IsOfKind(const ResultMessage& message);

	virtual void DoDestroy( void ) const;

	virtual Message& DoClone( id_type id ) const;

	static const Message::id_type BaseId;
};			

} } } } } // namespace axis::application::output::collectors::messages

