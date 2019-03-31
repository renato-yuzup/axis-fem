#include "SnapshotParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "foundation/ApplicationErrorException.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/jobs/AnalysisStep.hpp"

namespace aaj = axis::application::jobs;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;

namespace {
const int absolute_tag = 0;
const int relative_tag = 1;
const int range_tag = 2;
const int split_tag = 3;
} // namespace 

#ifdef AXIS_DOUBLEPRECISION
	static const real tolerance = 1e-14;
#else
	static const real tolerance = 1e-6;
#endif

aapps::SnapshotParser::SnapshotParser( bool ignoreSnapshotDeclarations )
{
	_ignoreSnapshotDeclarations = ignoreSnapshotDeclarations;
	_dirtySnapshotGroup = false;
	_lastSnapshotTime = 0;
	_canAddMoreSnapshots = true;

	InitGrammar();
}

aapps::SnapshotParser::~SnapshotParser( void )
{
	// nothing to do here
}

void aapps::SnapshotParser::InitGrammar( void )
{
	_absoluteMarkStatement	<< aslf::AxisGrammar::CreateOperatorParser(_T("SNAPSHOT"), absolute_tag) 
							            << aslf::AxisGrammar::CreateOperatorParser(_T("AT"))
							            << aslf::AxisGrammar::CreateNumberParser();

	_relativeMarkStatement	<< aslf::AxisGrammar::CreateOperatorParser(_T("SNAPSHOT"), relative_tag)
							            << aslf::AxisGrammar::CreateOperatorParser(_T("AGAIN"))
							            << aslf::AxisGrammar::CreateOperatorParser(_T("AFTER"))
							            << aslf::AxisGrammar::CreateNumberParser();

	_rangeMarkStatement		<< aslf::AxisGrammar::CreateOperatorParser(_T("SNAPSHOT"), range_tag)
							          << aslf::AxisGrammar::CreateOperatorParser(_T("EVERY"))
							          << aslf::AxisGrammar::CreateNumberParser()
							          << aslf::AxisGrammar::CreateOperatorParser(_T("FROM"))
							          << aslf::AxisGrammar::CreateNumberParser()
							          << aslf::AxisGrammar::CreateOperatorParser(_T("TO"))
							          << aslf::AxisGrammar::CreateNumberParser();

	_splitStatement			<< aslf::AxisGrammar::CreateOperatorParser(_T("DO"), split_tag)
							        << aslf::AxisGrammar::CreateNumberParser()
							        << aslf::AxisGrammar::CreateOperatorParser(_T("SNAPSHOTS"));

	_acceptedStatements << _absoluteMarkStatement 
						          << _relativeMarkStatement
						          << _rangeMarkStatement
						          << _splitStatement;
}

void aapps::SnapshotParser::DoCloseContext( void )
{
	// ignore marks if we have been told so or an error occurred
	if (_ignoreSnapshotDeclarations || _dirtySnapshotGroup)
	{
		return;
	}

	// add snapshots to timeline if the entire statement sequence
	// was correct
	aaj::StructuralAnalysis& analysis = GetAnalysis();
	ada::AnalysisTimeline& tl = GetParseContext().GetStepOnFocus()->GetTimeline();
	while(!_marks.empty())
	{
		tl.AddSnapshotMark(ada::SnapshotMark(_marks.front()));		
		_marks.pop_front();
	}
}

void aapps::SnapshotParser::DoStartContext( void )
{
	aaj::StructuralAnalysis& analysis = GetAnalysis();
	ada::AnalysisTimeline& tl = GetParseContext().GetStepOnFocus()->GetTimeline();
	_lastSnapshotTime = tl.StartTime();
}

aapps::BlockParser& aapps::SnapshotParser::GetNestedContext( const axis::String& contextName, 
                                                             const aslse::ParameterList& paramList )
{
	// we don't allow nested contexts
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::SnapshotParser::Parse( const asli::InputIterator& begin, 
                                                const asli::InputIterator& end )
{
	aslp::ParseResult result = _acceptedStatements(begin, end);

	if (result.IsMatch() && !_ignoreSnapshotDeclarations)
	{
		ParseSnapshotStatement(result.GetParseTree());
	}
	else if(result.GetResult() == aslp::ParseResult::FailedMatch)
	{
		// we cannot use this group to add snapshots anymore
		_dirtySnapshotGroup = true;
	}

	return result;
}

void aapps::SnapshotParser::ParseAbsoluteStatement( const aslp::ParseTreeNode& parseTree )
{
	const aslp::ParseTreeNode *timeMarkTerm = parseTree.GetNextSibling()->GetNextSibling();
	real timeValue = static_cast<const aslp::NumberTerminal *>(timeMarkTerm)->GetDouble();
	AddSnapshotMark(timeValue, false);
}

void aapps::SnapshotParser::ParseRelativeStatement( const aslp::ParseTreeNode& parseTree )
{
	const aslp::ParseTreeNode *timeMarkTerm = 
    parseTree.GetNextSibling()->GetNextSibling()->GetNextSibling();
	real timeValue = static_cast<const aslp::NumberTerminal *>(timeMarkTerm)->GetDouble();
	if (timeValue == 0)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_INVALID_SNAPSHOT_TIME, 
                                                       AXIS_ERROR_MSG_INVALID_SNAPSHOT_TIME));
		_dirtySnapshotGroup = true;
		return;
	}
	AddSnapshotMark(timeValue, true);
}

void aapps::SnapshotParser::ParseRangeStatement( const aslp::ParseTreeNode& parseTree )
{
	const aslp::ParseTreeNode *valueTerm = parseTree.GetNextSibling()->GetNextSibling();

	// the quantity...
	real splitAmount = static_cast<const aslp::NumberTerminal *>(valueTerm)->GetDouble();

	// the range start...
	valueTerm = valueTerm->GetNextSibling()->GetNextSibling();
	real rangeStart = static_cast<const aslp::NumberTerminal *>(valueTerm)->GetDouble();
	
	// ...and range end
	valueTerm = valueTerm->GetNextSibling()->GetNextSibling();
	real rangeEnd = static_cast<const aslp::NumberTerminal *>(valueTerm)->GetDouble();

	// check range correctness
	bool ok = true;
	if (rangeStart >= rangeEnd || splitAmount == 0)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_INVALID_SNAPSHOT_TIME, 
                                                       AXIS_ERROR_MSG_INVALID_SNAPSHOT_TIME));
		_dirtySnapshotGroup = true;
		ok = false;
	}

	long i = (long)((rangeEnd - rangeStart) / splitAmount);
	real t = rangeStart + splitAmount;
	for (; t <= rangeEnd; t += splitAmount)
	{
		if (!AddSnapshotMark(t, false))
		{	// abort
			return;
		}
	}
	t -= splitAmount;

	// if the last mark would exceed the given time range, place it
	// on the end of the range
	if ((rangeEnd - t) / splitAmount >= tolerance)	// that is, last mark did not fall exactly on range end
	{
		if (!AddSnapshotMark(rangeEnd, false))
		{	// abort
			return;
		}
	}
}

void aapps::SnapshotParser::ParseSplitStatement( const aslp::ParseTreeNode& parseTree )
{
	const aslp::NumberTerminal *amountTerm = 
    static_cast<const aslp::NumberTerminal *>(parseTree.GetNextSibling());

	if (!amountTerm->IsInteger())
	{	
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_INVALID_SNAPSHOT_TIME, 
                                                       AXIS_ERROR_MSG_INVALID_SNAPSHOT_TIME));
		_dirtySnapshotGroup = true;
		return;
	}

	// this statement must be unique
	if (_marks.size() != 0)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_SNAPSHOT_LOCKED, 
                                                       AXIS_ERROR_MSG_SNAPSHOT_LOCKED));
		_dirtySnapshotGroup = true;
	}

	// check amount correctness
	long markCount = amountTerm->GetInteger();
	bool ok = true;
	if (markCount <= 0)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_INVALID_SNAPSHOT_TIME, 
                                                       AXIS_ERROR_MSG_INVALID_SNAPSHOT_TIME));
		_dirtySnapshotGroup = true;
		ok = false;
	}

	aaj::StructuralAnalysis& analysis = GetAnalysis();
	ada::AnalysisTimeline& tl = GetParseContext().GetStepOnFocus()->GetTimeline();

	real timeWindow = tl.EndTime() - tl.StartTime();
	real increment = timeWindow / (real)markCount;
	real markTime = tl.StartTime() + increment;
	for (long i = 0; i < markCount; ++i, markTime += increment)
	{
		if (markTime > tl.EndTime())
		{
			markTime = tl.EndTime();
		}
		if (!AddSnapshotMark(markTime, false))
		{	// abort
			return;
		}
	}
	_lastSnapshotTime = tl.EndTime();
	_canAddMoreSnapshots = false;
}

void aapps::SnapshotParser::ParseSnapshotStatement( const aslp::ParseTreeNode& parseTree )
{
	// determine statement type by the associated value of the
	// first operator in the expression
	const aslp::ParseTreeNode *expression = 
    static_cast<const aslp::ExpressionNode&>(parseTree).GetFirstChild();
	const aslp::OperatorTerminal *op = static_cast<const aslp::OperatorTerminal *>(expression);

	switch(op->GetValue())
	{
	case absolute_tag:
		ParseAbsoluteStatement(*expression);
		break;
	case relative_tag:
		ParseRelativeStatement(*expression);
		break;
	case range_tag:
		ParseRangeStatement(*expression);
		break;
	case split_tag:
		ParseSplitStatement(*expression);
		break;
	default:
		// something wrong...
		throw axis::foundation::ApplicationErrorException(_T("Invalid snapshot operator value."));
	}
}

bool aapps::SnapshotParser::AddSnapshotMark( real timeValue, bool isRelative )
{
	if (!_canAddMoreSnapshots)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_SNAPSHOT_LOCKED, 
                                                       AXIS_ERROR_MSG_SNAPSHOT_LOCKED));
		return false;
	}
	aaj::StructuralAnalysis& analysis = GetAnalysis();
	ada::AnalysisTimeline& tl = GetParseContext().GetStepOnFocus()->GetTimeline();
	// calculate time mark
	real t = timeValue + (isRelative? _lastSnapshotTime : 0);
	// check for overlapping
	if (_lastSnapshotTime - t > tolerance)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_OVERLAPPED_SNAPSHOT, 
                                                       AXIS_ERROR_MSG_OVERLAPPED_SNAPSHOT));
		_dirtySnapshotGroup = true;
		return false;
	}
	//  check if time mark is valid
	if (tl.StartTime() - t > tolerance || t - tl.EndTime() > tolerance)
	{
		GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_SNAPSHOT_OUT_OF_RANGE,
                                                       AXIS_ERROR_MSG_SNAPSHOT_OUT_OF_RANGE));
		_dirtySnapshotGroup = true;
		return false;
	}
	// add snapshot
	_marks.push_back(t);
	_lastSnapshotTime = t;
	return true;
}
