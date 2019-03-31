#include "WaveSpeedProportionalClockworkFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "domain/algorithms/WaveSpeedProportionalClockwork.hpp"

namespace aafa = axis::application::factories::algorithms;
using namespace axis::services::language::syntax::evaluation;
using namespace axis::foundation;

const axis::String::char_type * aafa::WaveSpeedProportionalClockworkFactory::ClockworkTypeName = _T("EXPLICIT_TIME_CONTROL");

namespace {
const axis::String::char_type * dtimeScaleParamName = _T("DTIME_SCALE_FACTOR");
const axis::String::char_type * linearDTimeParamName = _T("DTIME_FORCE_LINEAR");
}

void axis::application::factories::algorithms::WaveSpeedProportionalClockworkFactory::Destroy( void ) const
{
	delete this;
}

bool axis::application::factories::algorithms::WaveSpeedProportionalClockworkFactory::CanBuild( const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime ) const
{
	int paramCount = 0;

	if (clockworkTypeName != ClockworkTypeName) return false;

	if (paramList.IsDeclared(dtimeScaleParamName))
	{
		ParameterValue& val = paramList.GetParameterValue(dtimeScaleParamName);
		if (!val.IsAtomic()) return false;
		AtomicValue& atomVal = static_cast<AtomicValue&>(val);
		if (!atomVal.IsNumeric()) return false;
		real scaleVal = static_cast<NumberValue&>(atomVal).GetDouble();
		if (scaleVal <= 0) return false;
		paramCount++;
	}
	if (paramList.IsDeclared(linearDTimeParamName))
	{
		ParameterValue& val = paramList.GetParameterValue(linearDTimeParamName);
		if (!val.IsAtomic()) return false;
		AtomicValue& atomVal = static_cast<AtomicValue&>(val);
		String linearVal = atomVal.ToString();
		if (linearVal != _T("YES") && linearVal != _T("NO") && linearVal != _T("TRUE") && linearVal != _T("FALSE"))
		{
			return false;
		}
		paramCount++;
	}

	if (paramList.Count() != paramCount) return false;
	return true;
}

axis::domain::algorithms::Clockwork& axis::application::factories::algorithms::WaveSpeedProportionalClockworkFactory::Build( const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime )
{
	if (!CanBuild(clockworkTypeName, paramList, stepStartTime, stepEndTime))
	{	// huh? that's funny...
		throw axis::foundation::InvalidOperationException(_T("Cannot build the specified object."));
	}

	real scaleFactor = 1;
	bool considerNonLinearity = true;

	if (paramList.IsDeclared(dtimeScaleParamName))
	{
		ParameterValue& val = paramList.GetParameterValue(dtimeScaleParamName);
		scaleFactor = static_cast<NumberValue&>(val).GetDouble();		
	}
	if (paramList.IsDeclared(linearDTimeParamName))
	{
		ParameterValue& val = paramList.GetParameterValue(linearDTimeParamName);
		String linearVal = val.ToString();
		considerNonLinearity = (linearVal == _T("NO") || linearVal == _T("FALSE"));
	}

	return *new axis::domain::algorithms::WaveSpeedProportionalClockwork(scaleFactor, considerNonLinearity);
}
