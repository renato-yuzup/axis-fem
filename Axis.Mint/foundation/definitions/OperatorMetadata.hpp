#pragma once
#include "foundation/Axis.Mint.hpp"
#include "OperatorInformation.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API OperatorMetadata
{
private:
	OperatorMetadata(void);
public:
	~OperatorMetadata(void);

	static const OperatorInformation OperatorAnd;
	static const OperatorInformation OperatorOr;
	static const OperatorInformation OperatorNot;
	static const OperatorInformation OpenGroup;
	static const OperatorInformation CloseGroup;

	friend class AxisInputLanguage;
};
			
} } } // namespace axis::foundation::definitions
