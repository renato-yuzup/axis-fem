#include "AxisInputLanguage.hpp"

namespace afd = axis::foundation::definitions;

afd::AxisInputLanguage::AxisInputLanguage( void )
{
	// nothing to do
}

afd::AxisInputLanguage::~AxisInputLanguage( void )
{
	// nothing to do
}

const axis::String::char_type * afd::AxisInputLanguage::EndBlockComment = _T("*/");
const axis::String::char_type * afd::AxisInputLanguage::BeginBlockComment = _T("/*");
const axis::String::char_type * afd::AxisInputLanguage::InlineComment = _T("#");

const afd::OperatorMetadata afd::AxisInputLanguage::Operators;

const afd::ElementBlockSyntax afd::AxisInputLanguage::ElementSyntax;
const afd::ElementSetBlockSyntax afd::AxisInputLanguage::ElementSetSyntax;
const afd::NodeSetBlockSyntax afd::AxisInputLanguage::NodeSetSyntax;
const afd::NodeBlockSyntax afd::AxisInputLanguage::NodeSyntax;
const afd::PartBlockSyntax afd::AxisInputLanguage::PartSyntax;
const afd::AnalysisSettingsBlockSyntax afd::AxisInputLanguage::AnalysisSettingsSyntax;
const afd::ResultCollectionBlockSyntax afd::AxisInputLanguage::ResultCollectionSyntax;
const afd::StepBlockSyntax afd::AxisInputLanguage::StepSyntax;

const axis::String::char_type * afd::AxisInputLanguage::LoadSectionBlockName = _T("LOADS");
const axis::String::char_type * afd::AxisInputLanguage::CurveSectionBlockName = _T("CURVES");
const axis::String::char_type * afd::AxisInputLanguage::ConstraintBlockName = _T("CONSTRAINTS");
const axis::String::char_type * afd::AxisInputLanguage::SnapshotsBlockName = _T("SNAPSHOTS");
