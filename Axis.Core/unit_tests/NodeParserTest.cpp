#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "application/parsing/parsers/NodeParser.hpp"
#include "application/factories/parsers/NodeParserProvider.hpp"
#include "services/language/syntax/evaluation/ParameterAssignment.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "application/parsing/core/ParseContextConcrete.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/parsing/error_messages.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "MockModuleManager.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "application/jobs/StructuralAnalysis.hpp"


using namespace axis;
using namespace axis::foundation;
using namespace axis::application::parsing::core;
using namespace axis::application::parsing::parsers;
using namespace axis::application::factories::parsers;
using namespace axis::services::language::syntax::evaluation;
using namespace axis::services::language::factories;
using namespace axis::services::language::iterators;
using namespace axis::services::language::parsing;
using namespace axis::domain::analyses;
using namespace axis::application::jobs;


// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const ParseResult::Result& v)
{
	return String::int_parse((long)v).data();
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const InputIterator& it)
{
	return std::wstring(1, *it);
}


namespace axis { namespace unit_tests { namespace core {


TEST_CLASS(NodeParserTest)
{
public:
  TEST_METHOD_INITIALIZE(SetUp)
  {
    axis::System::Initialize();
  }

  TEST_METHOD_CLEANUP(TearDown)
  {
    axis::System::Finalize();
  }

	TEST_METHOD(TestProviderCaps)
	{
		NodeParserProvider bnpp;
		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& emptyParams = ParameterList::Create();
		ParameterList& pl = ParameterList::Create();
		pl.AddParameter(_T("SET_ID"), id);

		// ensure that we can parse with empty param list
		Assert::AreEqual(true, bnpp.CanParse(_T("NODES"), emptyParams));

		// ensure that we can parse with
		Assert::AreEqual(true, bnpp.CanParse(_T("NODES"), pl));

		// ensure that we cannot parse these ones
		Assert::AreEqual(false, bnpp.CanParse(_T("BOGUS"), pl));
		Assert::AreEqual(false, bnpp.CanParse(_T("BOGUS"), emptyParams));

		delete &emptyParams;
		delete &pl;
	}

	TEST_METHOD(TestProviderTooMuchParams)
	{
		NodeParserProvider bnpp;
		IdValue &id = *new IdValue(_T("HELLO"));
		IdValue &bogusValue = *new IdValue(_T("BOGUS"));
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		pl.AddParameter(_T("SET_ID"), id);
		pl.AddParameter(_T("TEST"), bogusValue);

		// must NOT accept if there is too much params
		Assert::AreEqual(false, bnpp.CanParse(_T("NODES"), pl));

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODES"), pl);

		// once we activate context, an error must be thrown
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(AXIS_ERROR_ID_UNKNOWN_BLOCK_PARAM, 
                     (int)context.EventSummary().GetLastEventId());
		Logger::WriteMessage(_T("Last message logged in current context: "));
		Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

		ws.Destroy();
		delete &pl;
	}

	TEST_METHOD(TestSimpleParse)
	{
		NodeParserProvider bnpp;
		MockModuleManager mmm;
		bnpp.PostProcessRegistration(mmm);
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& emptyParams = ParameterList::Create();
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
    SymbolTable& st = context.Symbols();
		pl.AddParameter(_T("SET_ID"), id);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODES"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// try to parse this node...
		String testLine = _T("100 -0.52e2   \t\t  -4.5      0");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(true, analysis.ExistsNodeSet(_T("HELLO")));
		Assert::AreEqual(true, analysis.GetNodeSet(_T("HELLO")).Count() > 0);
		Assert::AreEqual(true, analysis.GetNodeSet(_T("HELLO")).GetByUserIndex(100).GetUserId() == 100);
		Assert::AreEqual((id_type)1L, analysis.Nodes().Count());

		// test if a symbol was defined in this context
		Assert::AreEqual(true, st.IsSymbolDefined(_T("100"), SymbolTable::kNode));

		delete &emptyParams;
		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestWrappedStatementParse)
	{
		NodeParserProvider bnpp;
		MockModuleManager mmm;
		bnpp.PostProcessRegistration(mmm);
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& emptyParams = ParameterList::Create();
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
    SymbolTable& st = context.Symbols();
		pl.AddParameter(_T("SET_ID"), id);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODES"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// try to parse this node...
		String testLine = _T("100 -0.52e2");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());

		testLine = _T("100 -0.52e2 +4e-2");
		begin = IteratorFactory::CreateStringIterator(testLine);
		end = IteratorFactory::CreateStringIterator(testLine.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());

		testLine = _T("100 -0.52e2 +4e-2 2");
		begin = IteratorFactory::CreateStringIterator(testLine);
		end = IteratorFactory::CreateStringIterator(testLine.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
		Assert::AreEqual(end, result.GetLastReadPosition());

		Assert::AreEqual(true, analysis.ExistsNodeSet(_T("HELLO")));
		Assert::AreEqual(true, analysis.GetNodeSet(_T("HELLO")).Count() > 0);
		Assert::AreEqual(true, analysis.GetNodeSet(_T("HELLO")).GetByUserIndex(100).GetUserId() == 100);
		Assert::AreEqual((id_type)1L, analysis.Nodes().Count());

		// test if a symbol was defined in this context
		Assert::AreEqual(true, st.IsSymbolDefined(_T("100"), SymbolTable::kNode));

		delete &emptyParams;
		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestParseOnBlockHeaderErrorCondition)
	{
		NodeParserProvider bnpp;
		MockModuleManager mmm;
		bnpp.PostProcessRegistration(mmm);
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& emptyParams = ParameterList::Create();
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
		pl.AddParameter(_T("BOGUS_PARAM"), id);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODES"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());	// should raise error: unrecognized param
		Assert::AreEqual(AXIS_ERROR_ID_UNKNOWN_BLOCK_PARAM, 
                     (int)context.EventSummary().GetLastEventId());
		// try to parse this node...
		String testLine = _T("100 -0.52e2   \t\t  -4.5      0");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);

		// even if we couldn't understand the header, we can still interpret nodes without problem...
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// ...however, we can't assure that we will create the nodes (or the node set)
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));
		Assert::AreEqual((id_type)0L, analysis.Nodes().Count());

		delete &emptyParams;
		delete &pl;
		ws.Destroy();
	}

};


} } }

#endif

