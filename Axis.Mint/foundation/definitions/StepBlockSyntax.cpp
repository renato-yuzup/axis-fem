#include "StepBlockSyntax.hpp"

namespace afd = axis::foundation::definitions;

afd::StepBlockSyntax::StepBlockSyntax( void )
{
	// nothing to do here
}


const axis::String::char_type * afd::StepBlockSyntax::BlockName = _T("STEP");
const axis::String::char_type * afd::StepBlockSyntax::StepEndTimeParameterName = _T("END_TIME");
const axis::String::char_type * afd::StepBlockSyntax::StepStartTimeParameterName = _T("START_TIME");
const axis::String::char_type * afd::StepBlockSyntax::StepTitleParameterName = _T("NAME");
const axis::String::char_type * afd::StepBlockSyntax::SolverTypeParameterName = _T("TYPE");
const axis::String::char_type * afd::StepBlockSyntax::ClockworkTypeParameterName = _T("DTIME_CONTROL_TYPE");
const axis::String::char_type * afd::StepBlockSyntax::ClockworkParamListParameterName = _T("DTIME_CONTROL_PARAMS");
