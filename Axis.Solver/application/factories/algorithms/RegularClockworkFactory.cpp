#include "RegularClockworkFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "domain/algorithms/RegularClockwork.hpp"

using namespace axis::services::language::syntax::evaluation;

const axis::String::char_type * axis::application::factories::algorithms::RegularClockworkFactory::ClockworkTypeName = _T("DTIME_CONSTANT");

namespace {
const axis::String::char_type * dtimeParamName = _T("DTIME");
}

void axis::application::factories::algorithms::RegularClockworkFactory::Destroy( void ) const
{
	delete this;
}

bool axis::application::factories::algorithms::RegularClockworkFactory::CanBuild( const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime ) const
{
	if (clockworkTypeName != ClockworkTypeName) return false;
	
	if (paramList.Count() != 1) return false;
	if (!paramList.IsDeclared(dtimeParamName)) return false;
	ParameterValue& val = paramList.GetParameterValue(dtimeParamName);
	if (!val.IsAtomic()) return false;
	AtomicValue& atomVal = static_cast<AtomicValue&>(val);
	if (!atomVal.IsNumeric()) return false;
	real dtimeVal = static_cast<NumberValue&>(atomVal).GetDouble();
	if (dtimeVal < 0) return false;

	return true;
}

axis::domain::algorithms::Clockwork& axis::application::factories::algorithms::RegularClockworkFactory::Build( const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime )
{
	if (!CanBuild(clockworkTypeName, paramList, stepStartTime, stepEndTime))
	{	// huh? that's funny...
		throw axis::foundation::InvalidOperationException(_T("Cannot build the specified object."));
	}
	ParameterValue& val = paramList.GetParameterValue(dtimeParamName);
	real dtimeVal = static_cast<NumberValue&>(val).GetDouble();
	return *new axis::domain::algorithms::RegularClockwork(dtimeVal);
}
