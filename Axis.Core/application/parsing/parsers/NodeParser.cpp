#include "NodeParser.hpp"
#include "AxisString.hpp"
#include "foundation/SyntaxMismatchException.hpp"
#include "foundation/CustomParserErrorException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/memory/pointer.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/error_messages.hpp"
#include "services/messaging/ErrorMessage.hpp"

namespace af = axis::foundation;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;
namespace aslf = axis::services::language::factories;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;
namespace afm = axis::foundation::memory;

aapps::NodeParser::NodeParser(aafp::BlockProvider& factory, aafe::NodeFactory& nodeFactory,
  const aslse::ParameterList& params) : _parentProvider(factory), 
  _nodeFactory(nodeFactory), _paramList(params.Clone())
{
	// initialize grammar
	_nodeExpression << aslf::AxisGrammar::CreateNumberParser() << _nodeIdSeparator <<
					           aslf::AxisGrammar::CreateNumberParser() << _nodeCoordSeparator <<
					           aslf::AxisGrammar::CreateNumberParser() << _nodeCoordSeparator <<
					           aslf::AxisGrammar::CreateNumberParser();
	_nodeIdSeparator << _nodeIdSeparatorType1 << _nodeIdSeparatorType2 << 
                      aslf::AxisGrammar::CreateBlankParser(true);
	_nodeIdSeparatorType1 << aslf::AxisGrammar::CreateBlankParser() << 
                           aslf::AxisGrammar::CreateOperatorParser(_T(":")) << 
                           aslf::AxisGrammar::CreateBlankParser();
	_nodeIdSeparatorType2 << aslf::AxisGrammar::CreateBlankParser() << 
                           aslf::AxisGrammar::CreateOperatorParser(_T(",")) << 
                           aslf::AxisGrammar::CreateBlankParser();
	_nodeCoordSeparator << _nodeIdSeparatorType2 << aslf::AxisGrammar::CreateBlankParser(true);
	_mustIgnoreParse = false;
	_currentNodeSet = NULL;
}

aapps::NodeParser::~NodeParser(void)
{
	delete &_paramList;
}

aapps::BlockParser& aapps::NodeParser::GetNestedContext( const axis::String& contextName, 
                                                         const aslse::ParameterList& paramList )
{
	// check if the parent provider knows any provider that can handle it
	if (!_parentProvider.ContainsProvider(contextName, paramList))
	{
		throw axis::foundation::NotSupportedException();
	}

	aafp::BlockProvider& subProvider = _parentProvider.GetProvider(contextName, paramList);
	return subProvider.BuildParser(contextName, paramList);
}

aslp::ParseResult aapps::NodeParser::Parse(const asli::InputIterator& begin, 
                                           const asli::InputIterator& end)
{
	// let's try to parse
	aapc::SymbolTable& st = GetParseContext().Symbols();
  aslp::ParseResult result = _nodeExpression(begin, end, false);
	ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	// check if we have a syntactically correct definition
	if (result.IsMatch())
	{
		// ok, extract node data
		id_type nodeId;
		double nodeX, nodeY, nodeZ;
		bool ok = ExtractAndValidateNodeData(result.GetParseTree(), nodeId, nodeX, nodeY, nodeZ);
		if (ok && !_mustIgnoreParse)
		{
			// create node
      afm::RelativePointer ptr = 
        _nodeFactory.CreateNode(nodeId, (coordtype)nodeX, (coordtype)nodeY, (coordtype)nodeZ);
			ade::Node &node = absref<ade::Node>(ptr);
			// try to add to list
			try
			{
				st.DefineOrRefreshSymbol(String::int_parse(nodeId), aapc::SymbolTable::kNode);
				model.Nodes().Add(ptr);
				if (_currentNodeSet != NULL)
				{
					_currentNodeSet->Add(ptr);
				}
			}
			catch (...)
			{	// duplicate node; ignore if we are running a last pass to solve cross-references
				delete &node;
	 			if (st.IsSymbolCurrentRoundDefined(String::int_parse(nodeId), aapc::SymbolTable::kNode))
	 			{	// this shouldn't have happened
					String s = AXIS_ERROR_MSG_NODE_PARSER_DUPLICATED_ID;
					s.append(String::int_parse(nodeId));
					GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300520, s));
	 			}
				else
				{	// it's ok, refresh symbol for this round
					st.DefineOrRefreshSymbol(String::int_parse(nodeId), aapc::SymbolTable::kNode);
				}
			}
		}
		else if(!ok)
		{	// type mismatch
			String s = AXIS_ERROR_MSG_INVALID_VALUE_TYPE;
			s.append(_T("one or more node parameters are invalid."));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
		}
	}
	return result;
}

void aapps::NodeParser::DoStartContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
	ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	// check parameter list correctness
	if (_paramList.IsDeclared(_T("SET_ID")))
	{
		aslse::ParameterValue& paramValue = _paramList.GetParameterValue(_T("SET_ID"));
		if (!paramValue.IsAtomic())
		{
			String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
			s.append(_T("SET_ID"));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
			_mustIgnoreParse = true;
		}
		aslse::AtomicValue& atomicVal = (aslse::AtomicValue&)paramValue;
		if (!(atomicVal.IsId() || atomicVal.IsString()))
		{
			if (atomicVal.IsNumeric())
			{	// it's ok only if it is a non-negative integer
				aslse::NumberValue& numVal = (aslse::NumberValue&)atomicVal;
				if (!(numVal.IsInteger() && numVal.GetLong() >= 0))
				{
					String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
					s.append(_T("SET_ID"));
					GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
					_mustIgnoreParse = true;
				}
			}
			else
			{
				String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
				s.append(_T("SET_ID"));
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
				_mustIgnoreParse = true;
			}
		}
		String nodeSetId = _paramList.GetParameterValue(_T("SET_ID")).ToString();
		if (st.IsSymbolCurrentRoundDefined(nodeSetId, aapc::SymbolTable::kNodeSet))
		{	// duplicated node set
			delete _currentNodeSet;
			String s = AXIS_ERROR_MSG_NODESET_PARSER_DUPLICATED_ID;
			s.append(nodeSetId);
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x30051B, s));
			_mustIgnoreParse = true;
		}

		// check if we are not in a new round and passing by the same
		// node set
		if (st.IsSymbolDefined(nodeSetId, aapc::SymbolTable::kNodeSet))
		{	// it is, use the existing set instead
			_currentNodeSet = &model.GetNodeSet(nodeSetId);
			_mustIgnoreParse = true;
		}
		else
		{	// no, it really is a first pass round
			_currentNodeSet = new adc::NodeSet();
			model.AddNodeSet(nodeSetId, *_currentNodeSet);
		}

		// define (or refresh) node set
		st.DefineOrRefreshSymbol(nodeSetId, aapc::SymbolTable::kNodeSet);

		_paramList.Consume(_T("SET_ID"));
	}
	// ignore anyways if we are not in trial mode
	if (GetParseContext().GetRunMode() != aapc::ParseContext::kTrialMode)
	{
		_mustIgnoreParse = true;
	}
	// if we still have parameters not processed, warn user
	if (_paramList.Count() > 0) _mustIgnoreParse = true;
	WarnUnrecognizedParams(_paramList);
}

bool aapps::NodeParser::ExtractAndValidateNodeData( aslp::ParseTreeNode& parseTree, 
  id_type& nodeId, double& nodeX, double& nodeY, double& nodeZ ) const
{
	// get parse nodes
	aslp::ExpressionNode& rootNode = (aslp::ExpressionNode&)parseTree;
	const aslp::NumberTerminal *nodeIdTerm = (const aslp::NumberTerminal *)rootNode.GetFirstChild();
	const aslp::NumberTerminal *nodeXTerm = nodeIdTerm->GetNextSibling()->IsTerminal()?
	  (const aslp::NumberTerminal*)nodeIdTerm->GetNextSibling() :
	  (const aslp::NumberTerminal*)nodeIdTerm->GetNextSibling()->GetNextSibling();
	const aslp::NumberTerminal *nodeYTerm = nodeXTerm->GetNextSibling()->IsTerminal()?
		(const aslp::NumberTerminal*)nodeXTerm->GetNextSibling() :
		(const aslp::NumberTerminal*)nodeXTerm->GetNextSibling()->GetNextSibling();
	const aslp::NumberTerminal *nodeZTerm = nodeYTerm->GetNextSibling()->IsTerminal()?
		(const aslp::NumberTerminal*)nodeYTerm->GetNextSibling() :
		(const aslp::NumberTerminal*)nodeYTerm->GetNextSibling()->GetNextSibling();
	// get attributes
	nodeId = nodeIdTerm->GetInteger();
	nodeX = nodeXTerm->GetDouble();
	nodeY = nodeYTerm->GetDouble();
	nodeZ = nodeZTerm->GetDouble();
	// validate attributes
	if (!nodeIdTerm->IsInteger() || nodeIdTerm->GetInteger() <= 0)
	{
		GetParseContext().RegisterEvent(
      asmm::ErrorMessage(0x300521, AXIS_ERROR_MSG_NODE_PARSER_INVALID_ID));
		return false;
	}
	return true;
}
