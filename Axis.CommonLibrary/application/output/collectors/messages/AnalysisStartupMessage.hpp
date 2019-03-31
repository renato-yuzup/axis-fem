#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "services/messaging/ResultMessage.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace messages {

class AXISCOMMONLIBRARY_API AnalysisStartupMessage : public axis::services::messaging::ResultMessage
{
public:
	AnalysisStartupMessage(void);
	AnalysisStartupMessage(const axis::String& description);
	virtual ~AnalysisStartupMessage(void);

	static bool IsOfKind(const ResultMessage& message);

	virtual void DoDestroy( void ) const;

	virtual Message& DoClone( id_type id ) const;

	static const Message::id_type BaseId;
};			

} } } } } // namespace axis::application::output::collectors::messages

