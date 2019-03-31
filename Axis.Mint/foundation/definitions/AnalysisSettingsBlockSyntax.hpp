#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API AnalysisSettingsBlockSyntax
{
public:
	static const axis::String::char_type * BlockName;
	friend class AxisInputLanguage;
private:
	AnalysisSettingsBlockSyntax(void);
};		

} } } // namespace axis::foundation::definitions
