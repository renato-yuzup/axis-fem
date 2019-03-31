#include "NodalLoadParser.hpp"

#include "foundation/NotSupportedException.hpp"

#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/IdTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"

#include "domain/analyses/NumericalModel.hpp"
#include "domain/boundary_conditions/VariableConstraint.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/elements/DoF.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "services/language/parsing/NumberTerminal.hpp"

namespace ada = axis::domain::analyses;
namespace adbc = axis::domain::boundary_conditions;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace adcv = axis::domain::curves;

namespace af = axis::foundation;
namespace afm = axis::foundation::memory;

namespace aslpp = axis::services::language::primitives;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asls = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;

namespace aafp = axis::application::factories::parsers;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aaj = axis::application::jobs;

aapps::NodalLoadParser::NodalLoadParser( aafp::BlockProvider& parentProvider ) :
_parentProvider(parentProvider)
{
	InitGrammar();
}

aapps::NodalLoadParser::~NodalLoadParser( void )
{
	// nothing to do here
}

void aapps::NodalLoadParser::InitGrammar( void )
{
	_directionExpression = new aslpp::EnumerationParser(_possibleDirections, true);

	// main expression
	_nodalLoadExpression << _completeNodalLoadExpression << _shortNodalLoadExpression;
	_shortNodalLoadExpression 
                << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"))
							  << aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
							  << _validIdentifiers
							  << aslf::AxisGrammar::CreateReservedWordParser(_T("BEHAVES"))
							  << aslf::AxisGrammar::CreateReservedWordParser(_T("AS"))
							  << _validIdentifiers
							  << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"))
							  << *_directionExpression
							  << _directionWord;
	_completeNodalLoadExpression 
                 << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"))
								 << aslf::AxisGrammar::CreateReservedWordParser(_T("SET"))
								 << _validIdentifiers
								 << aslf::AxisGrammar::CreateReservedWordParser(_T("BEHAVES"))
								 << aslf::AxisGrammar::CreateReservedWordParser(_T("AS"))
								 << _validIdentifiers
								 << aslf::AxisGrammar::CreateReservedWordParser(_T("ON"))
								 << *_directionExpression
								 << _directionWord
								 << aslf::AxisGrammar::CreateReservedWordParser(_T("SCALE"))
								 << aslf::AxisGrammar::CreateOperatorParser(_T("="))
								 << aslf::AxisGrammar::CreateNumberParser();

	// accept singular or plural writing
	_directionWord << aslf::AxisGrammar::CreateReservedWordParser(_T("DIRECTION")) 
                 << aslf::AxisGrammar::CreateReservedWordParser(_T("DIRECTIONS"));

	_validIdentifiers << aslf::AxisGrammar::CreateIdParser() 
                    << aslf::AxisGrammar::CreateNumberParser() 
                    << aslf::AxisGrammar::CreateStringParser();
	_possibleDirections << aslf::AxisGrammar::CreateOperatorParser(_T("ALL"), -1)
						          << aslf::AxisGrammar::CreateOperatorParser(_T("x"), 1)
						          << aslf::AxisGrammar::CreateOperatorParser(_T("y"), 2)
						          << aslf::AxisGrammar::CreateOperatorParser(_T("z"), 3)
						          << aslf::AxisGrammar::CreateOperatorParser(_T("X"), 1)
						          << aslf::AxisGrammar::CreateOperatorParser(_T("Y"), 2)
						          << aslf::AxisGrammar::CreateOperatorParser(_T("Z"), 3);
}

aapps::BlockParser& aapps::NodalLoadParser::GetNestedContext( const axis::String& contextName, const aslse::ParameterList& paramList )
{
	if (_parentProvider.ContainsProvider(contextName, paramList))
	{
		aafp::BlockProvider& provider = _parentProvider.GetProvider(contextName, paramList);
		BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
		nestedContext.SetAnalysis(GetAnalysis());
		return nestedContext;
	}

	// no provider found
	throw af::NotSupportedException();
}

aslp::ParseResult aapps::NodalLoadParser::Parse( const asli::InputIterator& begin, 
                                                 const asli::InputIterator& end )
{
	aslp::ParseResult result = _nodalLoadExpression(begin, end);

	if (result.IsMatch())
	{	// ok, interpret results and build load
		ParseNodalLoad(result.GetParseTree());
	}
	else if (result.GetResult() == aslp::ParseResult::FailedMatch)
	{
		// register invalid expression
		String s = AXIS_ERROR_MSG_INVALID_DECLARATION;
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
	}

	return result;
}

void aapps::NodalLoadParser::ParseNodalLoad( const aslp::ParseTreeNode& parseTree )
{
	String nodeSetId, curveId;
	bool directions[3];
	real scaleFactor;
	bool okToContinue = true;
	// read information from parse tree
	ReadParseInformation(parseTree, nodeSetId, curveId, directions, scaleFactor);
	// check if we have the node set and curve id
	aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	if (!model.ExistsNodeSet(nodeSetId))
	{	// node set not found
		okToContinue = false;
		if (GetParseContext().GetRunMode() != aapc::ParseContext::kInspectionMode)
		{	// mark node set as pending
			st.AddCurrentRoundUnresolvedSymbol(nodeSetId, aapc::SymbolTable::kNodeSet);
		}
		else
		{	// the item should already have been created; trigger an error
			String s = AXIS_ERROR_MSG_NODESET_NOT_FOUND;
			s += nodeSetId;
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300522, s));
		}
	}
	if (!model.Curves().Contains(curveId))
	{	// curve not found
		okToContinue = false;
		if (GetParseContext().GetRunMode() != aapc::ParseContext::kInspectionMode)
		{	// mark curve as pending
			st.AddCurrentRoundUnresolvedSymbol(curveId, aapc::SymbolTable::kCurve);
		}
		else if (GetParseContext().GetRunMode() == aapc::ParseContext::kInspectionMode)
		{	// the item should already have been created; trigger an error
			String s = AXIS_ERROR_MSG_CURVE_NOT_FOUND;
			s += curveId;
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300537, s));
		}
	}
	// check if all nodes in the node set were initialized
	if (!IsNodeSetInitialized(nodeSetId))
	{
		okToContinue = false;
	}
	// abort creation if we do not have sufficient data
	if (!okToContinue) return;
	adc::NodeSet& nodeSet = model.GetNodeSet(nodeSetId);
  afm::RelativePointer curvePtr = model.Curves().GetPointer(curveId);
	// check if we are not redefining loads
	for (int i = 0; i < 3; i++)
	{
		if (directions[i])
		{
			if (!CheckSymbols(nodeSetId, nodeSet, i, curveId))
			{	// we are redefining at least one element; abort
				return;
			}
		}
	}
	// create nodal loads on all specified directions
	for (int i = 0; i < 3; i++)
	{
		if (directions[i])
		{
			BuildNodalLoad(nodeSet, i, curvePtr, scaleFactor);
		}
	}
}

void aapps::NodalLoadParser::ReadParseInformation( const aslp::ParseTreeNode& parseTree, 
                                                   axis::String& nodeSetId, axis::String& curveId, 
                                                   bool *directionsEnabled, real& scaleFactor )
{
	for (int i = 0; i < 3; i++)
	{
		directionsEnabled[i] = false;
	}
	// first, get node set id
	aslp::ParseTreeNode *node = 
    ((aslp::ExpressionNode&)parseTree).GetFirstChild()->GetNextSibling()->GetNextSibling();
	nodeSetId = node->ToString();
	// now, the curve id
	node = node->GetNextSibling()->GetNextSibling()->GetNextSibling();
	curveId = node->ToString();

	// now, for the directions to apply
	node = node->GetNextSibling()->GetNextSibling();
	aslp::EnumerationExpression& enumeration = (aslp::EnumerationExpression&)*node;
	aslp::SymbolTerminal *terminal = (aslp::SymbolTerminal *)enumeration.GetFirstChild();
	while (terminal != NULL)
	{
		aslp::OperatorTerminal& directionOperator = (aslp::OperatorTerminal&)*terminal;
		int directionId = directionOperator.GetValue();
		if (directionId != -1)
		{	// it is a single direction
			directionsEnabled[directionId-1] = true;
		}
		else
		{	// apply to all directions
			directionsEnabled[0] = true;
			directionsEnabled[1] = true;
			directionsEnabled[2] = true;
		}
		terminal = (aslp::SymbolTerminal *)terminal->GetNextSibling();
	}
	// check if we have the optional scale parameter
	node = node->GetNextSibling()->GetNextSibling();
	if (node != NULL)
	{
		node = node->GetNextSibling()->GetNextSibling();
		scaleFactor = static_cast<aslp::NumberTerminal *>(node)->GetDouble();
	}
	else
	{
		scaleFactor = 1.0;
	}
}

bool aapps::NodalLoadParser::CheckSymbols( const axis::String& nodeSetId, adc::NodeSet& nodeSet, 
                                           int dofIndex, const axis::String& curveId )
{
	// search all nodes
  size_type count = nodeSet.Count();
  for (size_type i = 0; i < count; ++i)
	{
    aapc::SymbolTable& st = GetParseContext().Symbols();
    ade::Node& node = nodeSet.GetByPosition(i);
		String symbolName = GetSymbolName(node.GetUserId(), dofIndex);
    if (st.IsSymbolCurrentRoundDefined(symbolName, aapc::SymbolTable::kNodalBoundaryCondition))
		{	// someone already defined this symbol in the current read round
			String s = AXIS_ERROR_MSG_REDEFINED_SYMBOL;
			s += nodeSetId;
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300514, s));
			return false;
		}
	}
	// ok, no one defined these symbols
	return true;
}

void aapps::NodalLoadParser::BuildNodalLoad( adc::NodeSet& nodeSet, int dofIndex, 
                                             afm::RelativePointer& curvePtr, real scaleFactor )
{
	aapc::SymbolTable& st = GetParseContext().Symbols();
  aaj::AnalysisStep& step = *GetParseContext().GetStepOnFocus();
  size_type count = nodeSet.Count();
  for (size_type i = 0; i < count; ++i)
	{
    ade::Node& node = nodeSet.GetByPosition(i);
		String symbolName = GetSymbolName(node.GetUserId(), dofIndex);

		// attach nodal load to dof, if we already have not done it
		// (e.g. we are on a read round after the one we created it)
		ade::DoF& dof = node.GetDoF(dofIndex);
		
		if (!step.NodalLoads().Contains(dof))
		{
			adbc::BoundaryCondition& load = 
        *new adbc::VariableConstraint(adbc::BoundaryCondition::NodalLoad, curvePtr, scaleFactor);
			step.NodalLoads().Add(dof, load);
		}
		// add symbol
		st.DefineOrRefreshSymbol(symbolName, aapc::SymbolTable::kNodalBoundaryCondition);
	}
}

axis::String aapps::NodalLoadParser::GetSymbolName( id_type nodeId, int dofIndex ) const
{
	int stepIndex = GetParseContext().GetStepOnFocusIndex();
	return String::int_parse(nodeId) + _T("@@") + axis::String::int_parse((long)dofIndex) 
         + _T("@") + axis::String::int_parse((long)stepIndex);
}

bool aapps::NodalLoadParser::IsNodeSetInitialized( const axis::String& nodeSetId )
{
	aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
  bool ok = true;
	if (!model.ExistsNodeSet(nodeSetId)) return false;
	adc::NodeSet& nodeSet = model.GetNodeSet(nodeSetId);
  size_type count = nodeSet.Count();
  for (size_type i = 0; i < count; ++i)
	{
    ade::Node& node = nodeSet.GetByPosition(i);
		String nodeIdStr = String::int_parse(node.GetUserId());
		if (!st.IsSymbolDefined(nodeIdStr, aapc::SymbolTable::kNodeDof))
		{
			st.AddCurrentRoundUnresolvedSymbol(nodeIdStr, aapc::SymbolTable::kNodeDof);
			ok = false;
		}
	}
	return ok;
}
