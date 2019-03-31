#include "MultiLineCurveParser.hpp"

#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "foundation/NotSupportedException.hpp"
#include "foundation/memory/pointer.hpp"

#include "domain/analyses/NumericalModel.hpp"
#include "domain/curves/MultiLineCurve.hpp"

#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"

namespace aapps = axis::application::parsing::parsers;
namespace afm = axis::foundation::memory;
namespace adcu = axis::domain::curves;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;
namespace aapps = axis::application::parsing::parsers;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;

aapps::MultiLineCurveParser::MultiLineCurveParser( const axis::String& curveId )
{
	_curveId = curveId;
	_ignoreCurve = false;
	_onErrorRecovery = false;
	InitGrammar();
}
void aapps::MultiLineCurveParser::InitGrammar( void )
{
	/*
		We accept the following grammar:

		point_expression ::= grouped_expr | ungrouped_expr
		grouped_expr	 ::= '<' ungrouped_expr '>' | '(' ungrouped_expr ')' | '[' ungrouped_expr ']'
		grouped_expr	 ::= number separator number
		separator		 ::= whitespace<mandatory> | ';' | ','
	*/
	_valueSeparator         << aslf::AxisGrammar::CreateBlankParser(true) << _nonBlankValueSeparator;
	_nonBlankValueSeparator << aslf::AxisGrammar::CreateBlankParser() << _acceptedValueSeparator 
                          << aslf::AxisGrammar::CreateBlankParser();
	_acceptedValueSeparator << aslf::AxisGrammar::CreateOperatorParser(_T(",")) 
                          << aslf::AxisGrammar::CreateOperatorParser(_T(";"));

	_groupedExpression1	 << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateOperatorParser(_T("<")) 
                       << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateNumberParser() << _valueSeparator 
                       << aslf::AxisGrammar::CreateNumberParser() 
                       << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateOperatorParser(_T(">"));

	_groupedExpression2	 << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateOperatorParser(_T("(")) 
                       << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateNumberParser() << _valueSeparator 
                       << aslf::AxisGrammar::CreateNumberParser() 
                       << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateOperatorParser(_T(")"));

	_groupedExpression3	 << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateOperatorParser(_T("[")) 
                       << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateNumberParser() << _valueSeparator 
                       << aslf::AxisGrammar::CreateNumberParser() 
                       << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateOperatorParser(_T("]"));

	_ungroupedExpression << aslf::AxisGrammar::CreateBlankParser() 
                       << aslf::AxisGrammar::CreateNumberParser() << _valueSeparator 
                       << aslf::AxisGrammar::CreateNumberParser();

	_acceptedExpressions << _groupedExpression1 << _groupedExpression2 
                       << _groupedExpression3 << _ungroupedExpression;
}

aapps::MultiLineCurveParser::~MultiLineCurveParser( void )
{
	// nothing to do here
}

void aapps::MultiLineCurveParser::DoCloseContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	// ignore if on error recovery mode
	if (_onErrorRecovery) return;
	if (!_ignoreCurve)
	{
		// ok, so now we know all the points of the curve (and how many there are)
		size_t count = _points.size();
		size_t i = 0;
		// create the curve
    afm::RelativePointer curvePtr = adcu::MultiLineCurve::Create(count);
    adcu::MultiLineCurve& curve = absref<adcu::MultiLineCurve>(curvePtr);
		point_list::iterator end = _points.end();
		for (point_list::iterator it = _points.begin(); it != end; ++it)
		{
			curve.SetPoint(i, (real)it->x, (real)it->y);
			++i;
		}
		// add curve to model
		model.Curves().Add(_curveId, curvePtr);
	}
	st.DefineOrRefreshSymbol(_curveId, aapc::SymbolTable::kCurve);
}

void aapps::MultiLineCurveParser::DoStartContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
  _points.clear();

	// warn if we are redefining a curve
	if (st.IsSymbolCurrentRoundDefined(_curveId, aapc::SymbolTable::kCurve))
	{
		String s = AXIS_ERROR_MSG_REDEFINED_SYMBOL;
		s += _curveId;
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300514, s));
		_onErrorRecovery = true;
	}

	// do not process if we are redeclaring it on a new round
	_ignoreCurve = (model.Curves().Contains(_curveId));
}

aapps::BlockParser& aapps::MultiLineCurveParser::GetNestedContext(const axis::String& contextName, 
                                                                  const aslse::ParameterList& paramList)
{
	// we don't accept nested blocks
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::MultiLineCurveParser::Parse( const asli::InputIterator& begin, 
                                                      const asli::InputIterator& end )
{
	aslp::ParseResult result = _acceptedExpressions(begin, end, false);
	if (result.IsMatch())
	{	// syntax ok, add point to list, if not in error recovery mode
		if (!_onErrorRecovery)
		{
			Point p = ParsePointExpression(result.GetParseTree());
			_points.push_back(p);
		}
	}
	else if(result.GetResult() == aslp::ParseResult::FailedMatch)
	{	// syntax error; abort curve creation
		_onErrorRecovery = true;
		_points.clear();

		// register event
		axis::String s = AXIS_ERROR_MSG_INVALID_DECLARATION;
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300502, s));
	}

	return result;
}

aapps::MultiLineCurveParser::Point aapps::MultiLineCurveParser::ParsePointExpression( 
  const aslp::ParseTreeNode& parseTree )
{
	Point p;
	// we are assuming the parse tree is from a well-formed expression
	// first move to first node of the expression
	const aslp::ParseTreeNode *node = ((const aslp::ExpressionNode&)parseTree).GetFirstChild();
	// first, escape the open delimiter, if any
	if (((const aslp::SymbolTerminal *)node)->IsOperator())
	{
		node = node->GetNextSibling();
	}
	// read the x coordinate
	p.x = ((const aslp::NumberTerminal *)node)->GetDouble();
	// check if there is a separator; if yes, jump it
	node = node->GetNextSibling();
	if (!node->IsTerminal())
	{
		node = node->GetNextSibling();
	}
	else if (!((const aslp::SymbolTerminal *)node)->IsNumber())
	{
		node = node->GetNextSibling();
	}
	// read the y coordinate
	p.y = ((const aslp::NumberTerminal *)node)->GetDouble();
	return p;
}
