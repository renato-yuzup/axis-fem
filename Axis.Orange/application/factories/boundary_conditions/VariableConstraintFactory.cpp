#include "VariableConstraintFactory.hpp"

#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"
#include "services/language/parsing/ParseTreeNode.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "foundation/ArgumentException.hpp"

#include "application/jobs/AnalysisStep.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"

#include "domain/curves/Curve.hpp"
#include "domain/elements/Node.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"

#define SCALE_OP_IDENTIFIER       1
#define RELEASE_OP_IDENTIFIER     2

namespace aafbc = axis::application::factories::boundary_conditions;
namespace ada = axis::domain::analyses;
namespace af = axis::foundation;
namespace afm = axis::foundation::memory;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;
namespace asmm = axis::services::messaging;
namespace aapc = axis::application::parsing::core;
namespace aaj = axis::application::jobs;
namespace adc = axis::domain::collections;
namespace adcv = axis::domain::curves;
namespace ade = axis::domain::elements;
namespace adbc = axis::domain::boundary_conditions;

namespace {

  void ParseScaleFactorAndReleaseTime(real& scaleFactor, real& releaseTime, 
    const aslp::ParseTreeNode *node)
  {
    scaleFactor = 1;
    releaseTime = -1;

    if (node == nullptr) return;
    auto token = ((aslp::ExpressionNode *)node)->GetFirstChild();
    while (token != nullptr)
    {
      auto optExprNode = ((aslp::ExpressionNode *)token)->GetFirstChild();
      auto opNode = optExprNode->GetNextSibling();
      auto& op = static_cast<const aslp::OperatorTerminal&>(*opNode);
      if (op.GetValue() == SCALE_OP_IDENTIFIER) 
      { // a scale factor expression
        auto& scaleTerm = 
          static_cast<const aslp::NumberTerminal&>(*opNode->GetNextSibling());
        scaleFactor = scaleTerm.GetDouble();
      }
      else if (op.GetValue() == RELEASE_OP_IDENTIFIER)
      { // a release time expression
        auto& releaseTerm = 
          static_cast<const aslp::NumberTerminal&>(*opNode->GetNextSibling());
        releaseTime = releaseTerm.GetDouble();
      }
      token = token->GetNextSibling();
    }
  }
}


aafbc::VariableConstraintFactory::VariableConstraintFactory( 
  const axis::String& constraintStatementName )
{
	InitGrammar(constraintStatementName);
}

aafbc::VariableConstraintFactory::~VariableConstraintFactory( void )
{
	// nothing to do here
}

void aafbc::VariableConstraintFactory::InitGrammar( const axis::String& constraintStatementName )
{
	_dofExpression = new aslpp::EnumerationParser(_dofOperators, true);
	_shortVariableConstraintExpression << aslf::AxisGrammar::CreateReservedWordParser(_T("APPLY"))
		<< aslf::AxisGrammar::CreateReservedWordParser(constraintStatementName)
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("ON"))
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
		<< _idExpression
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("IN"))
		<< *_dofExpression
		<< _directionReservedWord
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("ACCORDING"))
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("TO"))
		<< _idExpression;

	_idExpression << aslf::AxisGrammar::CreateIdParser()
		<< aslf::AxisGrammar::CreateNumberParser()
		<< aslf::AxisGrammar::CreateStringParser();
	_dofOperators   << aslf::AxisGrammar::CreateOperatorParser(_T("X"), 0)
		<< aslf::AxisGrammar::CreateOperatorParser(_T("Y"), 1)
		<< aslf::AxisGrammar::CreateOperatorParser(_T("Z"), 2)
		<< aslf::AxisGrammar::CreateOperatorParser(_T("x"), 0)
		<< aslf::AxisGrammar::CreateOperatorParser(_T("y"), 1)
		<< aslf::AxisGrammar::CreateOperatorParser(_T("z"), 2)
		<< aslf::AxisGrammar::CreateOperatorParser(_T("ALL"), -1);
	_directionReservedWord << aslf::AxisGrammar::CreateReservedWordParser(_T("DIRECTION"))
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("DIRECTIONS"));

	// the full expression only adds the optional parameter 'SCALE'
  _scaleExpression
		<< aslf::AxisGrammar::CreateReservedWordParser(_T("SCALE"))
		<< aslf::AxisGrammar::CreateOperatorParser(_T("="), SCALE_OP_IDENTIFIER)
		<< aslf::AxisGrammar::CreateNumberParser();
  _releaseExpression
    << aslf::AxisGrammar::CreateReservedWordParser(_T("RELEASE"))
    << aslf::AxisGrammar::CreateOperatorParser(_T("AFTER"), RELEASE_OP_IDENTIFIER)
    << aslf::AxisGrammar::CreateNumberParser();
  _optExpr1 << _scaleExpression << _releaseExpression;
  _optExpr2 << _releaseExpression << _scaleExpression;
  _singleOptExpr1 << _scaleExpression;
  _singleOptExpr2 << _releaseExpression;
  _optionalExpressions << _optExpr1 << _optExpr2 
    << _singleOptExpr1 << _singleOptExpr2
    << aslf::AxisGrammar::CreateEpsilonParser();

	_variableConstraintExpression	<< _shortVariableConstraintExpression <<
    _optionalExpressions;
}

aafbc::VariableConstraintFactory::ParseParameters 
  aafbc::VariableConstraintFactory::ParseVariableConstraintStatement( const asli::InputIterator& begin, 
  const asli::InputIterator& end ) const
{
	ParseParameters result;	
	// try parse string
	result.ParsingResult = _variableConstraintExpression(begin, end);
	if (result.ParsingResult.IsMatch())
	{	// ok, let's get the information we need
		const aslp::ParseTreeNode *rootNode = &result.ParsingResult.GetParseTree();
    const aslp::ParseTreeNode *mainExprNode = ((aslp::ExpressionNode *)rootNode)->GetFirstChild();

    auto node = ((aslp::ExpressionNode *)mainExprNode)->GetFirstChild();
		node = node->GetNextSibling()->GetNextSibling()->GetNextSibling()->GetNextSibling();
		// node set id
		result.NodeSetId = node->ToString();
		// get directions
		const aslp::ExpressionNode *enumeration = 
      (aslp::EnumerationExpression *)node->GetNextSibling()->GetNextSibling();
		const aslp::SymbolTerminal *terminal = (aslp::SymbolTerminal *)enumeration->GetFirstChild();
		while (terminal != NULL)
		{
			aslp::OperatorTerminal& directionOp = (aslp::OperatorTerminal&)*terminal;
			switch (directionOp.GetValue())
			{
			case 0:
			case 1:
			case 2:
				result.Directions[directionOp.GetValue()] = true;
				break;
			default: // all directions
				result.Directions[0] = true;
				result.Directions[1] = true;
				result.Directions[2] = true;
			}
			terminal = (aslp::SymbolTerminal *)terminal->GetNextSibling();
		}
		// get curve id
		node = enumeration->GetNextSibling()->GetNextSibling()->GetNextSibling()->GetNextSibling();
		result.CurveId = node->ToString();

    // parse scale factor and release time, if exist
    ParseScaleFactorAndReleaseTime(result.ScalingFactor, 
      result.ReleaseTime, mainExprNode->GetNextSibling());
	}
	return result;
}

aslp::ParseResult aafbc::VariableConstraintFactory::TryParse( const asli::InputIterator& begin, 
                                                              const asli::InputIterator& end )
{
	ParseParameters result = ParseVariableConstraintStatement(begin, end);
	return result.ParsingResult;
}

aslp::ParseResult aafbc::VariableConstraintFactory::ParseAndBuild( aaj::StructuralAnalysis& analysis, 
                                                                   aapc::ParseContext& context, 
                                                                   const asli::InputIterator& begin, 
                                                                   const asli::InputIterator& end )
{
	ParseParameters result = ParseVariableConstraintStatement(begin, end);
	if (!result.ParsingResult.IsMatch())
	{	// uh oh, something is wrong...
		throw af::ArgumentException();
	}
	// first check for node set existence
	if (!IsNodeSetAvailable(result.NodeSetId, analysis.GetNumericalModel(), context)) 
    return result.ParsingResult;

	// now, check if we are not redefining boundary conditions in
	// the node's dof's
	if (!AreDofsAvailable(result.NodeSetId, result.Directions, analysis, context)) 
    return result.ParsingResult;

	// does the curve exist?
	if (!CurveExists(result.CurveId, analysis.GetNumericalModel(), context)) 
    return result.ParsingResult;	 

	// all clear, build constraints
	BuildVariableConstraint(result.NodeSetId, result.CurveId, result.Directions, 
                          result.ScalingFactor, result.ReleaseTime, 
                          analysis, context);
	return result.ParsingResult;
}

bool aafbc::VariableConstraintFactory::IsNodeSetAvailable( const axis::String& nodeSetId, 
                                                           ada::NumericalModel& analysis, 
                                                           aapc::ParseContext& context ) const
{
  aapc::SymbolTable& st = context.Symbols();
	bool isNodesInitialized = true;

	// check for node set
	if (!analysis.ExistsNodeSet(nodeSetId))
	{	// node set not found -- add unresolved reference
		if (context.GetRunMode() == aapc::ParseContext::kInspectionMode)
		{	// node set not created -- trigger an error
			String s = AXIS_ERROR_MSG_NODESET_NOT_FOUND;
			s += nodeSetId;
			context.RegisterEvent(asmm::ErrorMessage(0x300522, s));
		}
		else
		{	// tell that we need a node set
			st.AddCurrentRoundUnresolvedSymbol(nodeSetId, aapc::SymbolTable::kNodeSet);
		}
		return false;
	}

	// node set exists, but we also need to check if all nodes were
	// initialized (that is, has any dof)
	adc::NodeSet& nodeSet = analysis.GetNodeSet(nodeSetId);
  size_type count = nodeSet.Count();
  for (size_type i = 0; i < count; ++i)
	{
    ade::Node& node = nodeSet.GetByPosition(i);
		String nodeIdStr = String::int_parse(node.GetUserId());
		if (!st.IsSymbolDefined(nodeIdStr, aapc::SymbolTable::kNodeDof))
		{
			st.AddCurrentRoundUnresolvedSymbol(nodeIdStr, aapc::SymbolTable::kNodeDof);
			isNodesInitialized = false;
		}
	}
	return isNodesInitialized;
}

bool aafbc::VariableConstraintFactory::CurveExists( const axis::String& curveId, 
                                                    const ada::NumericalModel& model, 
                                                    aapc::ParseContext& context ) const
{
	// check for curve
	if (!model.Curves().Contains(curveId))
	{	// curve not found -- add unresolved reference
		if (context.GetRunMode() == aapc::ParseContext::kInspectionMode)
		{	// curve not created -- trigger an error
			String s = AXIS_ERROR_MSG_CURVE_NOT_FOUND;
			s += curveId;
			context.RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_CURVE_NOT_FOUND, s));
		}
		else
		{	// tell that we need a node set
      aapc::SymbolTable& st = context.Symbols();
			st.AddCurrentRoundUnresolvedSymbol(curveId, aapc::SymbolTable::kCurve);
		}
		return false;
	}
	return true;
}

bool aafbc::VariableConstraintFactory::AreDofsAvailable( const axis::String& nodeSetId, 
                                                         bool *enabledDirections, 
                                                         aaj::StructuralAnalysis& analysis, 
                                                         aapc::ParseContext& context ) const
{
  aapc::SymbolTable& st = context.Symbols();
	adc::NodeSet& nodeSet = analysis.GetNumericalModel().GetNodeSet(nodeSetId);
  size_type count = nodeSet.Count();
  for (size_type i = 0; i < count; ++i)
	{
		// get node
		ade::Node& node = nodeSet.GetByPosition(i);
		// check if someone already declared a boundary condition on
		// the specified dof's of that node
		for (int i = 0; i < 3; ++i)
		{
			if (enabledDirections[i])
			{
				// first, check if the node has already been initialized; if
				// not, wait until all dofs has been initialized
				if (node.GetDofCount() == 0)
				{
					return false;
				}
				// node dofs has been initialized; check for boundary
				// conditions
				String symbolName = 
          GetBoundaryConditionSymbolName(node.GetUserId(), i, context.GetStepOnFocusIndex());
				if (st.IsSymbolCurrentRoundDefined(symbolName, aapc::SymbolTable::kNodalBoundaryCondition))
				{	// uh oh...
					String s = AXIS_ERROR_MSG_BOUNDARY_CONDITION_REDEFINED;
					s = s.replace(_T("%1"), nodeSetId);
					context.RegisterEvent(asmm::ErrorMessage(0x300523, s));
					return false;
				}
			}
		}
	}
	// ok, everything is normal
	return true;
}

void aafbc::VariableConstraintFactory::BuildVariableConstraint( const axis::String& nodeSetId, 
                                                                const axis::String& curveId, 
                                                                bool *enabledDirections, 
                                                                real scaleFactor, 
                                                                real releaseTime,
                                                                aaj::StructuralAnalysis& analysis, 
                                                                aapc::ParseContext& context )
{
	aapc::SymbolTable& st = context.Symbols();
  aaj::AnalysisStep& step = *context.GetStepOnFocus();
	adc::NodeSet& nodeSet = analysis.GetNumericalModel().GetNodeSet(nodeSetId);
	afm::RelativePointer curvePtr = analysis.GetNumericalModel().Curves().GetPointer(curveId);
  size_type count = nodeSet.Count();
  for (size_type i = 0; i < count; ++i)
	{
		ade::Node& node = nodeSet.GetByPosition(i);

		for (int i = 0; i < 3; ++i)
		{
			if (enabledDirections[i])
			{
				ade::DoF& dof = node.GetDoF(i);
				// do not re-apply if we are re-processing the same char range
				if (!step.Displacements().Contains(dof))
				{	
					RegisterNewBoundaryCondition(step, dof, curvePtr, scaleFactor, releaseTime);
				}
				// define in the symbol table
				st.DefineOrRefreshSymbol(
          GetBoundaryConditionSymbolName(node.GetUserId(), i, context.GetStepOnFocusIndex()), 
					aapc::SymbolTable::kNodalBoundaryCondition);
			}
		}
	}
}

axis::String aafbc::VariableConstraintFactory::GetBoundaryConditionSymbolName( unsigned long nodeId, 
                                                                              int dofIndex, 
                                                                              int stepIndex ) const
{
	return String::int_parse(nodeId) + _T("@@") + String::int_parse((long)dofIndex) 
         + _T("@") + String::int_parse((long)stepIndex);
}

aafbc::VariableConstraintFactory::ParseParameters::ParseParameters( void )
{
	ParsingResult.SetResult(aslp::ParseResult::FailedMatch);
	Directions[0] = false; Directions[1] = false; Directions[2] = false;
	ScalingFactor = 1;
}
