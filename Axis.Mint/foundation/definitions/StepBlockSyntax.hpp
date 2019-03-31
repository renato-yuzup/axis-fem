#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"


namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API StepBlockSyntax
{
private:
	StepBlockSyntax(void);
public:
	static const axis::String::char_type * SolverTypeParameterName;
	static const axis::String::char_type * StepStartTimeParameterName;
	static const axis::String::char_type * StepEndTimeParameterName;
  static const axis::String::char_type * StepTitleParameterName;
	static const axis::String::char_type * BlockName;
	static const axis::String::char_type * ClockworkTypeParameterName;
	static const axis::String::char_type * ClockworkParamListParameterName;

	friend class AxisInputLanguage;
};

} } } // namespace axis::foundation::definitions
