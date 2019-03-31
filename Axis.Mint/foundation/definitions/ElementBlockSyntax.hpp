#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API ElementBlockSyntax
{
private:
	ElementBlockSyntax(void);
public:
	~ElementBlockSyntax(void);

	static const axis::String::char_type * BlockName;
	static const axis::String::char_type * SetIdAttributeName;

	friend class AxisInputLanguage;
};

} } } // namespace axis::foundation::definitions
