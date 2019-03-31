#include "GeneralElementCollectorParser.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/language/parsing/ReservedWordTerminal.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"
#include "services/language/parsing/NumberTerminal.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aafc = axis::application::factories::collectors;
namespace aslp = axis::services::language::parsing;
namespace aslf = axis::services::language::factories;
namespace aslpm = axis::services::language::primitives;
namespace asli = axis::services::language::iterators;

namespace {
  static const int SetExpressionIdentifier = -5;
  static const int ScaleExpressionIdentifier = -10;
  static const int ElementExpressionIdentifier = -15;
}


aafc::GeneralElementCollectorParser::GeneralElementCollectorParser( void )
{
  InitGrammar();
}

aafc::GeneralElementCollectorParser::~GeneralElementCollectorParser( void )
{
  delete direction3DEnum_;
  delete direction6DEnum_;
}

aafc::CollectorParseResult aafc::GeneralElementCollectorParser::Parse( const asli::InputIterator& begin, 
                                                                       const asli::InputIterator& end )
{
  aslp::ParseResult result = collectorStatement_(begin, end);
  if (result.IsMatch())
  {
    CollectorParseResult ecpr = InterpretParseTree(result);
    if (!ecpr.DoesActOnWholeModel() && ecpr.GetTargetSetName().empty())
    { // prohibited situation
      result.SetResult(aslp::ParseResult::FailedMatch);
    }
    return ecpr;
  }
  else if (result.GetResult() == aslp::ParseResult::FailedMatch)
  {
    result = scalarCollectorStatement_(begin, end);
    if (result.IsMatch())
    {
      CollectorParseResult ecpr = InterpretScalarParseTree(result);
      if (!ecpr.DoesActOnWholeModel() && ecpr.GetTargetSetName().empty())
      { // prohibited situation
        result.SetResult(aslp::ParseResult::FailedMatch);
      }
      return ecpr;
    }
  }
  return CollectorParseResult(result);
}

aafc::CollectorParseResult aafc::GeneralElementCollectorParser::InterpretParseTree( 
                                                        const aslp::ParseResult& result ) const
{
  // default values
  aafc::CollectorType collectorType = aafc::kUndefined;
  aaocs::SummaryType groupingType = aaocs::kNone;
  bool directionState[6] = {false, false, false, false, false, false};
  axis::String targetSetName;
  bool actOnWholeSet = true;
  real scaleFactor = (real)1.0;
  bool useScale = false;

  // go to starting node
  const aslp::ParseTreeNode& rootNode = result.GetParseTree();
  const aslp::ParseTreeNode * node = static_cast<const aslp::ExpressionNode&>(rootNode).GetFirstChild();

  // check if grouping was specified
  node = node->GetNextSibling();
  const aslp::SymbolTerminal& probableGroupingSymbol = static_cast<const aslp::SymbolTerminal&>(*node);
  if (probableGroupingSymbol.IsOperator())
  { // ok, it was specified
    groupingType = (aaocs::SummaryType)
      static_cast<const aslp::OperatorTerminal&>(probableGroupingSymbol).GetValue();
    node = node->GetNextSibling();
  }

  // get which collector type was specified
  node = node->GetNextSibling();
  const aslp::ParseTreeNode *typeExprNode = static_cast<const aslp::ExpressionNode&>(*node).GetFirstChild();
  const aslp::ReservedWordTerminal *typeNode = static_cast<const aslp::ReservedWordTerminal *>(typeExprNode);
  collectorType = (aafc::CollectorType)typeNode->GetValue();

  // check whether directions was specified
  typeExprNode = typeExprNode->GetNextSibling();
  if (typeExprNode != NULL)
  { // yes, it was
    if (static_cast<const aslp::ExpressionNode *>(typeExprNode)->IsEnumeration())
    { // a set of various directions (or maybe a single one besides 'ALL') was specified
      const aslp::EnumerationExpression *directionEnumExpr = 
        static_cast<const aslp::EnumerationExpression *>(typeExprNode);
      const aslp::ReservedWordTerminal *directionNode = static_cast<const aslp::ReservedWordTerminal *>(
        directionEnumExpr->GetFirstChild());
      while (directionNode != NULL)
      {
        int index = directionNode->GetValue();
        directionState[index] = true;
        directionNode = static_cast<const aslp::ReservedWordTerminal *>(directionNode->GetNextSibling());
      }
    }
    else
    { // 'ALL' was specified
      for (int i = 0; i < 6; ++i) directionState[i] = true;
    }
  }
  else
  { // nothing was specified; assume all directions
    for (int i = 0; i < 6; ++i) directionState[i] = true;
  }

  // check if a set was specified
  node = node->GetNextSibling();
  if (node != NULL)
  {
    const aslp::ExpressionNode& setExprNode = static_cast<const aslp::ExpressionNode&>(*node);
    const aslp::ParseTreeNode *setNode = 
      static_cast<const aslp::ParseTreeNode *>(setExprNode.GetFirstChild());
    bool isSetExprNode = 
      (static_cast<const aslp::ReservedWordTerminal *>(setNode)->GetValue() == SetExpressionIdentifier);
    if (isSetExprNode)
    {
      setNode = setNode->GetNextSibling()->GetNextSibling();
      actOnWholeSet = false;
      targetSetName = setNode->ToString();
    }

    // move to next node
    node = node->GetNextSibling();
  }

  // check if a scale factor was specified
  if (node != NULL)
  {
    const aslp::ExpressionNode& scaleExprNode = static_cast<const aslp::ExpressionNode&>(*node);
    const aslp::ParseTreeNode *scaleNode = 
      static_cast<const aslp::ParseTreeNode *>(scaleExprNode.GetFirstChild());
    bool isScaleExprNode = 
      (static_cast<const aslp::ReservedWordTerminal *>(scaleNode)->GetValue() == ScaleExpressionIdentifier);
    if (isScaleExprNode)
    {
      useScale = true;
      scaleNode = scaleNode->GetNextSibling()->GetNextSibling();
      scaleFactor = (real)static_cast<const aslp::NumberTerminal *>(scaleNode)->GetDouble();
    }
  }

  return CollectorParseResult(result, collectorType, groupingType, directionState, 
                              targetSetName, actOnWholeSet, scaleFactor, useScale);
}

aafc::CollectorParseResult aafc::GeneralElementCollectorParser::InterpretScalarParseTree( 
  const aslp::ParseResult& result ) const
{
  // default values
  aafc::CollectorType collectorType = aafc::kUndefined;
  aaocs::SummaryType groupingType = aaocs::kNone;
  axis::String targetSetName;
  bool actOnWholeSet = true;
  real scaleFactor = (real)1.0;
  bool useScale = false;

  // go to starting node
  const aslp::ParseTreeNode& rootNode = result.GetParseTree();
  const aslp::ParseTreeNode * node = static_cast<const aslp::ExpressionNode&>(rootNode).GetFirstChild();

  // check if grouping was specified
  node = node->GetNextSibling();
  const aslp::SymbolTerminal& probableGroupingSymbol = static_cast<const aslp::SymbolTerminal&>(*node);
  if (probableGroupingSymbol.IsOperator())
  { // ok, it was specified
    groupingType = (aaocs::SummaryType)
      static_cast<const aslp::OperatorTerminal&>(probableGroupingSymbol).GetValue();
    node = node->GetNextSibling();
  }

  // get which collector type was specified
  node = node->GetNextSibling();
  const aslp::ReservedWordTerminal *typeNode = static_cast<const aslp::ReservedWordTerminal *>(node);
  collectorType = (aafc::CollectorType)typeNode->GetValue();

  // check if a set was specified
  node = node->GetNextSibling();
  if (node != NULL)
  {
    const aslp::ExpressionNode& setExprNode = static_cast<const aslp::ExpressionNode&>(*node);
    const aslp::ParseTreeNode *setNode = 
      static_cast<const aslp::ParseTreeNode *>(setExprNode.GetFirstChild());
    bool isSetExprNode = 
      (static_cast<const aslp::ReservedWordTerminal *>(setNode)->GetValue() == SetExpressionIdentifier);
    if (isSetExprNode)
    {
      setNode = setNode->GetNextSibling()->GetNextSibling();
      actOnWholeSet = false;
      targetSetName = setNode->ToString();
    }

    // move to next node
    node = node->GetNextSibling();
  }

  // check if a scale factor was specified
  if (node != NULL)
  {
    const aslp::ExpressionNode& scaleExprNode = static_cast<const aslp::ExpressionNode&>(*node);
    const aslp::ParseTreeNode *scaleNode = 
      static_cast<const aslp::ParseTreeNode *>(scaleExprNode.GetFirstChild());
    bool isScaleExprNode = 
      (static_cast<const aslp::ReservedWordTerminal *>(scaleNode)->GetValue() == ScaleExpressionIdentifier);
    if (isScaleExprNode)
    {
      useScale = true;
      scaleNode = scaleNode->GetNextSibling()->GetNextSibling();
      scaleFactor = (real)static_cast<const aslp::NumberTerminal *>(scaleNode)->GetDouble();
    }
  }

  return CollectorParseResult(result, collectorType, groupingType, nullptr, 
    targetSetName, actOnWholeSet, scaleFactor, useScale);
}

void axis::application::factories::collectors::GeneralElementCollectorParser::InitGrammar( void )
{
  anyIdentifierExpression_ << aslf::AxisGrammar::CreateIdParser() 
    << aslf::AxisGrammar::CreateStringParser(false);

  groupingType_ << aslf::AxisGrammar::CreateOperatorParser(_T("AVERAGE"), (int)aaocs::kAverage)
    << aslf::AxisGrammar::CreateOperatorParser(_T("MAXIMUM"), (int)aaocs::kMaximum)
    << aslf::AxisGrammar::CreateOperatorParser(_T("MINIMUM"), (int)aaocs::kMinimum)
    << aslf::AxisGrammar::CreateEpsilonParser();
  reactionForceName_ << aslf::AxisGrammar::CreateReservedWordParser(_T("REACTION"), (int)aafc::kReactionForce) 
    << aslf::AxisGrammar::CreateReservedWordParser(_T("FORCE"));
  collectorType3D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("DISPLACEMENT"), (int)aafc::kDisplacement)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("VELOCITY"), (int)aafc::kVelocity)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ACCELERATION"), (int)aafc::kAcceleration)
    << reactionForceName_;
  collectorType6D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("STRESS"), (int)aafc::kStress)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("STRAIN"), (int)aafc::kStrain)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("PLASTIC_STRAIN"), (int)aafc::kPlasticStrain);
  direction3D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("X"), 0)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("Y"), 1)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("Z"), 2);
  direction6D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("XX"), 0)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("YY"), 1)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ZZ"), 2)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("YZ"), 3)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("XZ"), 4)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("XY"), 5);

  direction3DEnum_ = new axis::services::language::primitives::EnumerationParser(direction3D_, true);
  direction6DEnum_ = new axis::services::language::primitives::EnumerationParser(direction6D_, true);

  optionalDirection3DExpression_ << *direction3DEnum_ 
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ALL"), -1)
    << aslf::AxisGrammar::CreateEpsilonParser();
  collectorType3DExpression_ << collectorType3D_ << optionalDirection3DExpression_;

  optionalDirection6DExpression_ << *direction6DEnum_ 
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ALL"), -1)
    << aslf::AxisGrammar::CreateEpsilonParser();
  collectorType6DExpression_ << collectorType6D_ << optionalDirection6DExpression_;

  /*
  * TODO: 3D Collectors were hidden because appropriate finite element calculation must be 
  * implemented first!
  **/
  collectorTypeExpression_ << collectorType6DExpression_; //  << collectorType3DExpression_;

  setExpression_ << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"), SetExpressionIdentifier)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
    << anyIdentifierExpression_;
  optionalSetExpression_ << setExpression_ << aslf::AxisGrammar::CreateEpsilonParser();

  scaleExpression_ << aslf::AxisGrammar::CreateReservedWordParser(_T("SCALE"), ScaleExpressionIdentifier)
    << aslf::AxisGrammar::CreateOperatorParser(_T("="))
    << aslf::AxisGrammar::CreateNumberParser();
  optionalScaleExpression_ << scaleExpression_ << aslf::AxisGrammar::CreateEpsilonParser();

  collectorStatement_ << aslf::AxisGrammar::CreateReservedWordParser(_T("RECORD")) 
    << groupingType_
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ELEMENT"), ElementExpressionIdentifier)
    << collectorTypeExpression_
    << optionalSetExpression_
    << optionalScaleExpression_;

  // these rules are specific to scalar quantities
  scalarGroupingType_ << aslf::AxisGrammar::CreateOperatorParser(_T("AVERAGE"), (int)aaocs::kAverage)
                      << aslf::AxisGrammar::CreateOperatorParser(_T("MAXIMUM"), (int)aaocs::kMaximum)
                      << aslf::AxisGrammar::CreateOperatorParser(_T("MINIMUM"), (int)aaocs::kMinimum)
                      << aslf::AxisGrammar::CreateOperatorParser(_T("TOTAL"), (int)aaocs::kSum)
                      << aslf::AxisGrammar::CreateEpsilonParser();
  scalarCollectorType_ << aslf::AxisGrammar::CreateReservedWordParser(_T("ARTIFICIAL_ENERGY"), (int)aafc::kArtificialEnergy)
     << aslf::AxisGrammar::CreateReservedWordParser(_T("EFFECTIVE_PLASTIC_STRAIN"), (int)aafc::kEffectivePlasticStrain)
     << aslf::AxisGrammar::CreateReservedWordParser(_T("DEFORMATION_GRADIENT"), (int)aafc::kDeformationGradient);
  scalarCollectorStatement_ << aslf::AxisGrammar::CreateReservedWordParser(_T("RECORD")) 
    << scalarGroupingType_
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ELEMENT"), ElementExpressionIdentifier)
    << scalarCollectorType_
    << optionalSetExpression_
    << optionalScaleExpression_;
}
