#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API PartBlockSyntax
{
private:
	PartBlockSyntax(void);
public:
	static const axis::String::char_type * ElementTypeParameterName;
	static const axis::String::char_type * ElementDescriptionParameterName;
	static const axis::String::char_type * BlockName;
				
	friend class AxisInputLanguage;
};

} } } // namespace axis::foundation::definitions
