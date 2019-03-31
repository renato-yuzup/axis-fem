#pragma once
#include "services/messaging/filters/EventLogMessageFilter.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace filters {		

class AXISCOMMONLIBRARY_API AnalysisEventFilter : public axis::services::messaging::filters::EventLogMessageFilter
{
private:
	bool _isFiltering;
public:
	AnalysisEventFilter(void);
	~AnalysisEventFilter(void);

	virtual bool IsEventMessageFiltered( const axis::services::messaging::EventMessage& message );

	virtual bool IsResultMessageFiltered( const axis::services::messaging::ResultMessage& message );

	virtual void Destroy( void ) const;

	virtual MessageFilter& Clone( void ) const;
};

} } } } } // namespace axis::application::output::collectors::filters

