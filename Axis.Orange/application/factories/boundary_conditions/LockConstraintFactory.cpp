#include "LockConstraintFactory.hpp"

#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"
#include "services/language/parsing/ParseTreeNode.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "foundation/ArgumentException.hpp"

#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"

#include "domain/elements/Node.hpp"
#include "domain/boundary_conditions/LockConstraint.hpp"
#include "application/jobs/AnalysisStep.hpp"


namespace af = axis::foundation;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;
namespace asmm = axis::services::messaging;
namespace aaj = axis::application::jobs;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace adbc = axis::domain::boundary_conditions;
namespace aafbc = axis::application::factories::boundary_conditions;

aafbc::LockConstraintFactory::LockConstraintFactory( void )
{
	InitGrammar();
}

aafbc::LockConstraintFactory::~LockConstraintFactory( void )
{
	// nothing to do here
}

void aafbc::LockConstraintFactory::InitGrammar( void )
{
	_dofExpression = new aslpp::EnumerationParser(_dofOperators, true);
	_constraintExpression << aslf::AxisGrammar::CreateReservedWordParser(_T("LOCK"))
						            << aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
						            << _nodeSetIdExpression
						            << aslf::AxisGrammar::CreateReservedWordParser(_T("IN"))
						            << *_dofExpression
						            << _directionReservedWord
                        << _optionalExpr;

	_nodeSetIdExpression << aslf::AxisGrammar::CreateIdParser()
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
  _releaseExpr << aslf::AxisGrammar::CreateReservedWordParser(_T("RELEASE"))
               << aslf::AxisGrammar::CreateOperatorParser(_T("AFTER"))
               << aslf::AxisGrammar::CreateNumberParser();
  _optionalExpr << _releaseExpr << aslf::AxisGrammar::CreateEpsilonParser();

}

aafbc::LockConstraintFactory::ParseParameters aafbc::LockConstraintFactory::ParseConstraint(
  const asli::InputIterator& begin, const asli::InputIterator& end ) const
{
	ParseParameters result;

	// try parse string
	result.ParsingResult = _constraintExpression(begin, end);
	if (result.ParsingResult.IsMatch())
	{	// ok, let's get the information we need
		const aslp::ParseTreeNode *node = &result.ParsingResult.GetParseTree();
		node = ((aslp::ExpressionNode *)node)->GetFirstChild();
		node = node->GetNextSibling()->GetNextSibling();

		// id node
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

    // check if release time was specified
    result.ReleaseTime = -1;
    const aslp::ParseTreeNode *releaseNode = 
      enumeration->GetNextSibling()->GetNextSibling();
    if (releaseNode != nullptr)
    {
      auto exprNode = static_cast<const aslp::ExpressionNode *>(releaseNode);
      auto firstTerm = exprNode->GetFirstChild();

      const aslp::NumberTerminal& releaseTerm = 
        static_cast<const aslp::NumberTerminal&>(
        *firstTerm->GetNextSibling()->GetNextSibling());
      result.ReleaseTime = releaseTerm.GetDouble();
    }
	}

	return result;
}

aslp::ParseResult aafbc::LockConstraintFactory::TryParse( const asli::InputIterator& begin, 
                                                          const asli::InputIterator& end )
{
	ParseParameters result = ParseConstraint(begin, end);
	return result.ParsingResult;
}

aslp::ParseResult aafbc::LockConstraintFactory::ParseAndBuild( aaj::StructuralAnalysis& analysis, 
                                                               aapc::ParseContext& context, 
                                                               const asli::InputIterator& begin, 
                                                               const asli::InputIterator& end )
{
	ParseParameters result = ParseConstraint(begin, end);
	if (!result.ParsingResult.IsMatch())
	{	// uh oh, something is wrong...
		throw af::ArgumentException();
	}

	// first check for node set existence
	if (!NodeSetExists(result.NodeSetId, analysis.GetNumericalModel(), context)) 
    return result.ParsingResult;

	// now, check if we are not redefining boundary conditions in
	// the node's dof's
	if (!CheckForRedefinition(result.NodeSetId, result.Directions, analysis, context)) 
    return result.ParsingResult;

	// all clear, build constraints
	BuildClampingConstraints(result.NodeSetId, result.Directions, 
    result.ReleaseTime, analysis, context);

	return result.ParsingResult;
}

bool aafbc::LockConstraintFactory::NodeSetExists( const axis::String& nodeSetId, 
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

bool aafbc::LockConstraintFactory::CheckForRedefinition( const axis::String& nodeSetId, 
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
				String symbolName = GetBoundaryConditionSymbolName(node.GetUserId(), i, context.GetStepOnFocusIndex());
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

void aafbc::LockConstraintFactory::BuildClampingConstraints( const axis::String& nodeSetId, 
                                                             bool *enabledDirections, 
                                                             real releaseTime,
                                                             aaj::StructuralAnalysis& analysis, 
                                                             aapc::ParseContext& context ) const
{
	aapc::SymbolTable& st = context.Symbols();
	ada::NumericalModel& model = analysis.GetNumericalModel();
	adc::NodeSet& nodeSet = model.GetNodeSet(nodeSetId);
	aaj::AnalysisStep& step = *context.GetStepOnFocus();

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
				ade::DoF& dof = node[i];
				if (!step.Locks().Contains(dof))
				{	// do not re-apply if we are re-processing the same char range
					adbc::BoundaryCondition& c = *new adbc::LockConstraint(releaseTime);
					step.Locks().Add(dof, c);
				}

				// define in the symbol table
				st.DefineOrRefreshSymbol(
            GetBoundaryConditionSymbolName(node.GetUserId(), i, context.GetStepOnFocusIndex()), 
						aapc::SymbolTable::kNodalBoundaryCondition);
			}
		}
	}
}

axis::String aafbc::LockConstraintFactory::GetBoundaryConditionSymbolName( unsigned long nodeId, 
                                                                           int dofIndex, 
                                                                           int stepIndex ) const
{
	return String::int_parse(nodeId) + _T("@@") + 
         String::int_parse((long)dofIndex) + _T("@") + String::int_parse((long)stepIndex);
}

aafbc::LockConstraintFactory::ParseParameters::ParseParameters( void )
{
	ParsingResult.SetResult(aslp::ParseResult::FailedMatch);
	Directions[0] = false;
	Directions[1] = false;
	Directions[2] = false;
}
