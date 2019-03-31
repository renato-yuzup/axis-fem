#pragma once
#include "services/messaging/filters/MessageFilter.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace filters {		

class AXISCOMMONLIBRARY_API SingleResultMessageFilter : public axis::services::messaging::filters::MessageFilter
{
private:
	axis::services::messaging::Message::id_type _messageId;
public:
	SingleResultMessageFilter(axis::services::messaging::Message::id_type resultMessageId);
	~SingleResultMessageFilter(void);

	virtual bool IsEventMessageFiltered( const axis::services::messaging::EventMessage& message );

	virtual bool IsResultMessageFiltered( const axis::services::messaging::ResultMessage& message );

	virtual void Destroy( void ) const;

	virtual MessageFilter& Clone( void ) const;
};

} } } } } // namespace axis::application::output::collectors::filters
