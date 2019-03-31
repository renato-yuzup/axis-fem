#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"
#include "AnalysisSettingsBlockSyntax.hpp"
#include "ElementBlockSyntax.hpp"
#include "ElementSetBlockSyntax.hpp"
#include "NodeBlockSyntax.hpp"
#include "NodeSetBlockSyntax.hpp"
#include "OperatorMetadata.hpp"
#include "PartBlockSyntax.hpp"
#include "ResultCollectionBlockSyntax.hpp"
#include "StepBlockSyntax.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API AxisInputLanguage
{
private:
	AxisInputLanguage(void);
public:
	~AxisInputLanguage(void);

	static const axis::String::char_type * InlineComment;
	static const axis::String::char_type * BeginBlockComment;
	static const axis::String::char_type * EndBlockComment;

	static const OperatorMetadata Operators;

	static const PartBlockSyntax PartSyntax;
	static const NodeBlockSyntax NodeSyntax;
	static const NodeSetBlockSyntax NodeSetSyntax;
	static const ElementSetBlockSyntax ElementSetSyntax;
	static const ElementBlockSyntax ElementSyntax;
	static const AnalysisSettingsBlockSyntax AnalysisSettingsSyntax;
	static const ResultCollectionBlockSyntax ResultCollectionSyntax;
	static const StepBlockSyntax StepSyntax;

	static const axis::String::char_type * LoadSectionBlockName;
	static const axis::String::char_type * CurveSectionBlockName;
	static const axis::String::char_type * ConstraintBlockName;
	static const axis::String::char_type * SnapshotsBlockName;
};
		
} } } // namespace axis::foundation::definitions
