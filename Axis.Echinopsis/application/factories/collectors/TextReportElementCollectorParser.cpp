#include "TextReportElementCollectorParser.hpp"
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
  static const int NodeExpressionIdentifier = -15;
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult::ElementCollectorParseResult( 
  const aslp::ParseResult& result ) :
parseResult_(result), collectorType_(kUndefined)
{
  bool d[6] = {true, true, true, true, true, true};
  Init(d);
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult::ElementCollectorParseResult( 
  const aslp::ParseResult& result, 
  ElementCollectorParseResult::CollectorType collectorType, 
  const bool *directionState, 
  const axis::String& targetSetName ) :
parseResult_(result), collectorType_(collectorType), targetSetName_(targetSetName)
{
  Init(directionState);
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult::ElementCollectorParseResult( 
  const ElementCollectorParseResult& other )
{
  operator =(other);
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult& 
  aafc::TextReportElementCollectorParser::ElementCollectorParseResult::operator=( 
  const ElementCollectorParseResult& other )
{
  parseResult_ = other.parseResult_;
  collectorType_ = other.collectorType_;
  targetSetName_ = other.targetSetName_;
  Init(other.directionState_);
  return *this;
}

void aafc::TextReportElementCollectorParser::ElementCollectorParseResult::Init(
  const bool * directionState )
{
  switch (collectorType_)
  {
  case kStress: 
  case kStrain:
    directionCount_ = 6;
    break;
  default:
    directionCount_ = 3;
    break;
  }
  for (int i = 0; i < directionCount_; ++i)
  {
    directionState_[i] = directionState[i];
  }
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult::~ElementCollectorParseResult( void )
{
  // nothing to do here
}

aslp::ParseResult aafc::TextReportElementCollectorParser::ElementCollectorParseResult::GetParseResult( void ) const
{
  return parseResult_;
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult::CollectorType 
  aafc::TextReportElementCollectorParser::ElementCollectorParseResult::GetCollectorType( void ) const
{
  return collectorType_;
}

axis::String aafc::TextReportElementCollectorParser::ElementCollectorParseResult::GetTargetSetName( void ) const
{
  return targetSetName_;
}

bool aafc::TextReportElementCollectorParser::ElementCollectorParseResult::ShouldCollectDirection( int directionIndex ) const
{
  switch (collectorType_)
  {
  case kStress:
  case kStrain:
    if (directionIndex < 0 || directionIndex >= 6)
    {
      throw axis::foundation::OutOfBoundsException();
    }
    break;
  default:
    if (directionIndex < 0 || directionIndex >= 3)
    {
      throw axis::foundation::OutOfBoundsException();
    }
    break;
  }
  return directionState_[directionIndex];
}

aafc::TextReportElementCollectorParser::TextReportElementCollectorParser( void )
{
  InitGrammar();
}

aafc::TextReportElementCollectorParser::~TextReportElementCollectorParser( void )
{
  delete direction3DEnum_;
  delete direction6DEnum_;
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult 
  aafc::TextReportElementCollectorParser::Parse( const asli::InputIterator& begin, 
  const asli::InputIterator& end )
{
  aslp::ParseResult result = collectorStatement_(begin, end);
  if (result.IsMatch())
  {
    ElementCollectorParseResult ncpr = InterpretParseTree(result);
    if (ncpr.GetTargetSetName().empty())
    { // prohibited situation
      result.SetResult(aslp::ParseResult::FailedMatch);
    }
    return ncpr;
  }
  return ElementCollectorParseResult(result);
}

aafc::TextReportElementCollectorParser::ElementCollectorParseResult 
  aafc::TextReportElementCollectorParser::InterpretParseTree( const aslp::ParseResult& result ) const
{
  // default values
  ElementCollectorParseResult::CollectorType collectorType = ElementCollectorParseResult::kUndefined;
  bool directionState[6] = {false, false, false, false, false, false};
  axis::String targetSetName;

  // go to starting node
  const aslp::ParseTreeNode& rootNode = result.GetParseTree();
  const aslp::ParseTreeNode * node = static_cast<const aslp::ExpressionNode&>(rootNode).GetFirstChild();

  // get which collector type was specified
  node = node->GetNextSibling()->GetNextSibling();
  const aslp::ParseTreeNode *typeExprNode = static_cast<const aslp::ExpressionNode&>(*node).GetFirstChild();
  const aslp::ReservedWordTerminal *typeNode = static_cast<const aslp::ReservedWordTerminal *>(typeExprNode);
  collectorType = (ElementCollectorParseResult::CollectorType)typeNode->GetValue();

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

  node = node->GetNextSibling();
  const aslp::ExpressionNode& setExprNode = static_cast<const aslp::ExpressionNode&>(*node);
  const aslp::ParseTreeNode *setNode = 
    static_cast<const aslp::ParseTreeNode *>(setExprNode.GetFirstChild());
  setNode = setNode->GetNextSibling()->GetNextSibling();
  targetSetName = setNode->ToString();

  return ElementCollectorParseResult(result, collectorType, directionState, targetSetName);
}

void axis::application::factories::collectors::TextReportElementCollectorParser::InitGrammar( void )
{
  anyIdentifierExpression_ << aslf::AxisGrammar::CreateIdParser() 
    << aslf::AxisGrammar::CreateStringParser(false);

  collectorType6D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("STRESS"), (int)ElementCollectorParseResult::kStress)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("STRAIN"), (int)ElementCollectorParseResult::kStrain);
  direction6D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("XX"), 0)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("YY"), 1)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ZZ"), 2)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("YZ"), 3)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("XZ"), 4)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("XY"), 5);

  direction6DEnum_ = new axis::services::language::primitives::EnumerationParser(direction6D_, true);

  optionalDirection6DExpression_ << *direction6DEnum_ 
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ALL"), -1)
    << aslf::AxisGrammar::CreateEpsilonParser();
  collectorType6DExpression_ << collectorType6D_ << optionalDirection6DExpression_;

  collectorTypeExpression_ << collectorType6DExpression_;

  setExpression_ << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"), SetExpressionIdentifier)
    << aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
    << anyIdentifierExpression_;

  collectorStatement_ << aslf::AxisGrammar::CreateReservedWordParser(_T("RECORD")) 
    << aslf::AxisGrammar::CreateReservedWordParser(_T("ELEMENT"), NodeExpressionIdentifier)
    << collectorTypeExpression_
    << setExpression_;
}
