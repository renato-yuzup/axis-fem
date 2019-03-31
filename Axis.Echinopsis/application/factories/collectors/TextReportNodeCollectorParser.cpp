#include "TextReportNodeCollectorParser.hpp"
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

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::NodeCollectorParseResult( 
                                                        const aslp::ParseResult& result ) :
parseResult_(result), collectorType_(kUndefined)
{
  bool d[6] = {true, true, true, true, true, true};
  Init(d);
}

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::NodeCollectorParseResult( 
                                                        const aslp::ParseResult& result, 
                                                        NodeCollectorParseResult::CollectorType collectorType, 
                                                        const bool *directionState, 
                                                        const axis::String& targetSetName ) :
parseResult_(result), collectorType_(collectorType), targetSetName_(targetSetName)
{
  Init(directionState);
}

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::NodeCollectorParseResult( 
                                                        const NodeCollectorParseResult& other )
{
  operator =(other);
}

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult& 
    aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::operator=( 
    const NodeCollectorParseResult& other )
{
  parseResult_ = other.parseResult_;
  collectorType_ = other.collectorType_;
  targetSetName_ = other.targetSetName_;
  Init(other.directionState_);
  return *this;
}

void aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::Init(
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

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::~NodeCollectorParseResult( void )
{
  // nothing to do here
}

aslp::ParseResult aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::GetParseResult( void ) const
{
  return parseResult_;
}

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::CollectorType 
      aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::GetCollectorType( void ) const
{
  return collectorType_;
}

axis::String aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::GetTargetSetName( void ) const
{
  return targetSetName_;
}

bool aafc::TextReportNodeCollectorParser::NodeCollectorParseResult::ShouldCollectDirection( int directionIndex ) const
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

aafc::TextReportNodeCollectorParser::TextReportNodeCollectorParser( void )
{
  InitGrammar();
}

aafc::TextReportNodeCollectorParser::~TextReportNodeCollectorParser( void )
{
  delete direction3DEnum_;
  delete direction6DEnum_;
}

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult 
  aafc::TextReportNodeCollectorParser::Parse( const asli::InputIterator& begin, 
                                              const asli::InputIterator& end )
{
  aslp::ParseResult result = collectorStatement_(begin, end);
  if (result.IsMatch())
  {
    NodeCollectorParseResult ncpr = InterpretParseTree(result);
    if (ncpr.GetTargetSetName().empty())
    { // prohibited situation
      result.SetResult(aslp::ParseResult::FailedMatch);
    }
    return ncpr;
  }
  return NodeCollectorParseResult(result);
}

aafc::TextReportNodeCollectorParser::NodeCollectorParseResult 
  aafc::TextReportNodeCollectorParser::InterpretParseTree( const aslp::ParseResult& result ) const
{
  // default values
  NodeCollectorParseResult::CollectorType collectorType = NodeCollectorParseResult::kUndefined;
  bool directionState[6] = {false, false, false, false, false, false};
  axis::String targetSetName;
  
  // go to starting node
  const aslp::ParseTreeNode& rootNode = result.GetParseTree();
  const aslp::ParseTreeNode * node = static_cast<const aslp::ExpressionNode&>(rootNode).GetFirstChild();

  // get which collector type was specified
  node = node->GetNextSibling()->GetNextSibling();
  const aslp::ParseTreeNode *typeExprNode = static_cast<const aslp::ExpressionNode&>(*node).GetFirstChild();
  const aslp::ReservedWordTerminal *typeNode = static_cast<const aslp::ReservedWordTerminal *>(typeExprNode);
  collectorType = (NodeCollectorParseResult::CollectorType)typeNode->GetValue();

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

  return NodeCollectorParseResult(result, collectorType, directionState, targetSetName);
}

void axis::application::factories::collectors::TextReportNodeCollectorParser::InitGrammar( void )
{
  anyIdentifierExpression_ << aslf::AxisGrammar::CreateIdParser() 
                           << aslf::AxisGrammar::CreateStringParser(false);

  collectorType3D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("DISPLACEMENT"), (int)NodeCollectorParseResult::kDisplacement)
                   << aslf::AxisGrammar::CreateReservedWordParser(_T("VELOCITY"), (int)NodeCollectorParseResult::kVelocity)
                   << aslf::AxisGrammar::CreateReservedWordParser(_T("ACCELERATION"), (int)NodeCollectorParseResult::kAcceleration)
                   << aslf::AxisGrammar::CreateReservedWordParser(_T("LOAD"), (int)NodeCollectorParseResult::kExternalLoad)
                   << aslf::AxisGrammar::CreateReservedWordParser(_T("REACTION"), (int)NodeCollectorParseResult::kReactionForce) ;
  collectorType6D_ << aslf::AxisGrammar::CreateReservedWordParser(_T("STRESS"), (int)NodeCollectorParseResult::kStress)
                   << aslf::AxisGrammar::CreateReservedWordParser(_T("STRAIN"), (int)NodeCollectorParseResult::kStrain);
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

  collectorTypeExpression_ << collectorType3DExpression_ << collectorType6DExpression_;

  setExpression_ << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"), SetExpressionIdentifier)
                 << aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
                 << anyIdentifierExpression_;

  collectorStatement_ << aslf::AxisGrammar::CreateReservedWordParser(_T("RECORD")) 
                      << aslf::AxisGrammar::CreateReservedWordParser(_T("NODAL"), NodeExpressionIdentifier)
                      << collectorTypeExpression_
                      << setExpression_;
}
