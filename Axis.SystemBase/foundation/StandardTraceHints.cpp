#include "StandardTraceHints.hpp"

using namespace axis::foundation;

axis::foundation::StandardTraceHints::StandardTraceHints( void )
{
	// nothing to do
}

axis::foundation::StandardTraceHints::~StandardTraceHints( void )
{
	// nothing to do
}

/* Initialize read-only static members */
const SourceTraceHint& axis::foundation::StandardTraceHints::AnalysisBlockReaderLogic		= SourceTraceHint(10001);
const SourceTraceHint& axis::foundation::StandardTraceHints::InputStreamFormatter			= SourceTraceHint(10002);
const SourceTraceHint& axis::foundation::StandardTraceHints::ModuleManagerControl			= SourceTraceHint(10003);
const SourceTraceHint& axis::foundation::StandardTraceHints::PreProcessorControl			= SourceTraceHint(10004);