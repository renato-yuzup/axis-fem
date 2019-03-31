#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "application/factories/parsers/NodeSetParserProvider.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "application/parsing/core/ParseContextConcrete.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/error_messages.hpp"
#include "MockModuleManager.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "foundation/memory/pointer.hpp"


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
using namespace axis::domain::elements;
using namespace axis::application::jobs;
using namespace axis::foundation::memory;

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

TEST_CLASS(NodeSetParserTest)
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
		NodeSetParserProvider bnpp;
		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& pl = ParameterList::Create();
		pl.AddParameter(_T("ID"), id);

		// ensure that we can parse with this set of params
		Assert::AreEqual(true, bnpp.CanParse(_T("NODE_SET"), pl));

		// ensure that we cannot parse this one
		Assert::AreEqual(false, bnpp.CanParse(_T("BOGUS"), pl));

		delete &pl;
	}

	TEST_METHOD(TestProviderInsufficientParams)
	{
		NodeSetParserProvider bnpp;
		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& emptyParams = ParameterList::Create();

		// ensure that we cannot parse with this set of params
		Assert::AreEqual(false, bnpp.CanParse(_T("NODE_SET"), emptyParams));
		delete &emptyParams;
	}

	TEST_METHOD(TestProviderTooMuchParams)
	{
		NodeSetParserProvider bnpp;
		IdValue &id = *new IdValue(_T("HELLO"));
		IdValue &bogusValue = *new IdValue(_T("BOGUS"));
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		pl.AddParameter(_T("ID"), id);
		pl.AddParameter(_T("TEST"), bogusValue);

		// must NOT accept if there is too much params
		Assert::AreEqual(false, bnpp.CanParse(_T("NODE_SET"), pl));

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODE_SET"), pl);
		parser.SetAnalysis(ws);

		// once we activate context, an error must be thrown
		parser.StartContext(context);
		Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());
		Assert::AreEqual(AXIS_ERROR_ID_UNKNOWN_BLOCK_PARAM, (int)context.EventSummary().GetLastEventId());
		Logger::WriteMessage(_T("Last message logged in current context: "));
		Logger::WriteMessage(context.EventSummary().GetLastEvent().GetDescription());

		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestSimpleParse)
	{
		NodeSetParserProvider bnpp;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
		pl.AddParameter(_T("ID"), id);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODE_SET"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// check node set existence; must not have added yet
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));

		// try to parse this range...
		String testLine = _T("10");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// try to parse this range too
		testLine = _T("  11 - 13");
		begin = IteratorFactory::CreateStringIterator(testLine);
		end = IteratorFactory::CreateStringIterator(testLine.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// closing the context must not add the node set to the
		// analysis object as there are unresolved references
		parser.CloseContext();
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));

		delete &parser;
		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestCrossRefParse)
	{
		NodeSetParserProvider bnpp;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
    SymbolTable& st = context.Symbols();
		pl.AddParameter(_T("ID"), id);

		// add a node to the analysis object
		RelativePointer ptr1 = Node::Create(10, 10);
		Node *node = absptr<Node>(ptr1);
		node->InitDofs(3,0);
		analysis.Nodes().Add(ptr1);

		// tell that we have initialized the node
		st.DefineOrRefreshSymbol(_T("10"), SymbolTable::kNodeDof);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODE_SET"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// node set must have not been added yet
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));

		// try to parse this range...
		String testLine = _T("10");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// must have found the node
		Assert::AreEqual(false, st.IsSymbolCurrentRoundUnresolved(_T("10"), SymbolTable::kNode));

		// try to parse this range too
		testLine = _T("  11 - 13");
		begin = IteratorFactory::CreateStringIterator(testLine);
		end = IteratorFactory::CreateStringIterator(testLine.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// none of these nodes exist; should have added a cross-ref
    Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("11"), SymbolTable::kNode));
    Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("12"), SymbolTable::kNode));
    Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("13"), SymbolTable::kNode));

		// closing the context must not add the incomplete node set to
		// the analysis object
		parser.CloseContext();
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));

		delete &parser;
		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestCrossRefResolvedParse)
	{
		NodeSetParserProvider bnpp;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
    SymbolTable& st = context.Symbols();
		pl.AddParameter(_T("ID"), id);

		// add a node to the analysis object
	  RelativePointer ptr1 = Node::Create(10, 10);
    Node *node = absptr<Node>(ptr1);
		node->InitDofs(3, 0);
		analysis.Nodes().Add(ptr1);

		// tell that we have initialized the node
		st.DefineOrRefreshSymbol(_T("10"), SymbolTable::kNodeDof);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODE_SET"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// node set must have not been added yet
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));

		// try to parse this range...
		String testLine = _T("10");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		// must have found the node
		Assert::AreEqual(false, st.IsSymbolCurrentRoundDefined(_T("10"), SymbolTable::kNode));

		// closing the context must add the node set to the analysis
		// object as it is complete
		parser.CloseContext();
		Assert::AreEqual(true, analysis.ExistsNodeSet(_T("HELLO")));

		// check if node set is ok
		Assert::AreEqual(true, analysis.GetNodeSet(_T("HELLO")).IsUserIndexed(10));

		// test if a symbol was defined in this context
		Assert::AreEqual(true, st.IsSymbolDefined(_T("HELLO"), SymbolTable::kNodeSet));

		delete &parser;
		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestWrappedStatementParse)
	{
		NodeSetParserProvider bnpp;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
		ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
		pl.AddParameter(_T("ID"), id);

		// add a node to the analysis object
		RelativePointer ptr1 = Node::Create(10, 3);
    Node *node = absptr<Node>(ptr1);
		analysis.Nodes().Add(ptr1);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODE_SET"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);

		// try to parse this range...
		String testLine = _T("10 -");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		testLine = _T("10 - 12");
		begin = IteratorFactory::CreateStringIterator(testLine);
		end = IteratorFactory::CreateStringIterator(testLine.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(true, !context.EventSummary().HasAnyEventRegistered());

		delete &pl;
		ws.Destroy();
	}

	TEST_METHOD(TestParseOnBlockHeaderErrorCondition)
	{
		NodeSetParserProvider bnpp;
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		IdValue &id = *new IdValue(_T("HELLO"));
    IdValue &id2 = *new IdValue(_T("TEST"));
    ParameterList& pl = ParameterList::Create();
		ParseContextConcrete context;
    SymbolTable& st = context.Symbols();
		pl.AddParameter(_T("BOGUS"), id);
    pl.AddParameter(_T("ID"), id2);

		// add a node to the analysis object
    RelativePointer ptr1 = Node::Create(10, 3);
		Node *node = absptr<Node>(ptr1);
		analysis.Nodes().Add(ptr1);

		// get parser
		BlockParser& parser = bnpp.BuildParser(_T("NODE_SET"), pl);
		parser.SetAnalysis(ws);
		parser.StartContext(context);

		// should warn about illegal parameter
		Assert::AreEqual(true, context.EventSummary().HasAnyEventRegistered());

		// check node set existence
		Assert::AreEqual(false, analysis.ExistsNodeSet(_T("HELLO")));

		// try to parse this range...
		String testLine = _T("10");
		InputIterator begin = IteratorFactory::CreateStringIterator(testLine);
		InputIterator end = IteratorFactory::CreateStringIterator(testLine.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// only the first event must have been registered
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());

		// try to parse this range too
		testLine = _T("  11 - 13");
		begin = IteratorFactory::CreateStringIterator(testLine);
		end = IteratorFactory::CreateStringIterator(testLine.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());

		// none of these nodes exist; should have added a cross-ref, even on error
		// condition
    Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("11"), SymbolTable::kNode));
    Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("12"), SymbolTable::kNode));
    Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("13"), SymbolTable::kNode));

		delete &pl;
		ws.Destroy();
	}

};


} } }


#endif

