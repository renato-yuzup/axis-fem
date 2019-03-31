#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "application/locators/MaterialFactoryLocator.hpp"
#include "application/parsing/core/ParseContextConcrete.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "application/parsing/parsers/PartParser.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/evaluation/NullValue.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/language/syntax/evaluation/ArrayValue.hpp"
#include "services/language/syntax/evaluation/ParameterAssignment.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "AxisString.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "application/parsing/core/Sketchbook.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "MockProviders.hpp"
#include "application/parsing/core/SymbolTable.hpp"


using namespace axis;
using namespace axis::foundation;
using namespace axis::application::factories::parsers;
using namespace axis::application::locators;
using namespace axis::application::parsing::core;
using namespace axis::application::parsing::parsers;
using namespace axis::domain::materials;
using namespace axis::services::language::syntax::evaluation;
using namespace axis::services::language::factories;
using namespace axis::services::language::iterators;
using namespace axis::services::language::parsing;
using namespace axis::foundation;
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

/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
TEST_CLASS(PartParserTest)
{
private:
  typedef PartParserTemplate<MockElementProvider, MockMaterialProvider> MockedPartParser;
public:
  TEST_METHOD_INITIALIZE(SetUp)
  {
    axis::System::Initialize();
  }

  TEST_METHOD_CLEANUP(TearDown)
  {
    axis::System::Finalize();
  }

	TEST_METHOD(TestConstructor)
	{
		// build our mock objects
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build parameter list, should accept anything at this point
		ParameterList &emptyParam = ParameterList::Create();
		ParameterList &param1 = ParameterList::Create();
		param1.AddParameter(_T("BOGUS"), *new NullValue());

		MockedPartParser pp1(parentProvider, elementProvider, materialProvider, emptyParam);
		MockedPartParser pp2(parentProvider, elementProvider, materialProvider, param1);

		emptyParam.Destroy();
		param1.Destroy();
	}

	TEST_METHOD(TestValidationInsufficientSectionArgs)
	{
		// build our mock objects
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// first, let's try to build with an empty param list
		ParameterList &emptyParam = ParameterList::Create();
		MockedPartParser pp1(parentProvider, elementProvider, materialProvider, emptyParam);
		pp1.StartContext(context);
		Assert::AreEqual(2, (int)context.EventSummary().GetTotalEventCount());	// two events because it couldn't find element type and its properties
		context.ClearEventStatistics();
		emptyParam.Destroy();

		// now, let's try to build with an insufficient param list
		ParameterList& paramList1 = ParameterList::Create();
		paramList1.AddParameter(_T("ELEM_TYPE"),*new IdValue(_T("BOGUS")));
		MockedPartParser pp2(parentProvider, elementProvider, materialProvider, paramList1);
		pp2.StartContext(context);
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		context.ClearEventStatistics();
		paramList1.Destroy();

		ParameterList& paramList2 = ParameterList::Create();
		paramList1.AddParameter(_T("PROPERTIES"),*new NullValue());
		MockedPartParser pp3(parentProvider, elementProvider, materialProvider, paramList1);
		pp3.StartContext(context);
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		context.ClearEventStatistics();
		paramList2.Destroy();
	}

	TEST_METHOD(TestValidationNoElementFound)
	{
		// build our mock objects
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// let's try to build with elem type as a bogus data
		ParameterList& paramList1 = ParameterList::Create();
		paramList1.AddParameter(_T("ELEM_TYPE"), *new NumberValue(0.0));
		paramList1.AddParameter(_T("PROPERTIES"), *new NullValue());
		MockedPartParser pp1(parentProvider, elementProvider, materialProvider, paramList1);
		pp1.StartContext(context);
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(1, elementProvider.GetFailedElementBuildQueryCount());
		Assert::AreEqual(0, elementProvider.GetSuccessfulElementBuildQueryCount());
		context.ClearEventStatistics();
		elementProvider.ResetCounters();
		paramList1.Destroy();

		// ...and then without the expected parameters
		ParameterList& paramList2 = ParameterList::Create();
		paramList2.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("BOGUS"), *new NullValue()));
		paramList2.AddParameter(_T("PROPERTIES"), *properties);
		MockedPartParser pp2(parentProvider, elementProvider, materialProvider, paramList2);
		pp2.StartContext(context);
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(1, elementProvider.GetFailedElementBuildQueryCount());
		Assert::AreEqual(0, elementProvider.GetSuccessfulElementBuildQueryCount());
		context.ClearEventStatistics();
		elementProvider.ResetCounters();
		paramList2.Destroy();
	}

	TEST_METHOD(TestValidateSectionArgsPass)
	{
		// build our mock objects
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build our parameter list
		ParameterList& paramList = ParameterList::Create();
		paramList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("MY_PROP"), *new NullValue()));
		paramList.AddParameter(_T("PROPERTIES"), *properties);

		MockedPartParser pp(parentProvider, elementProvider, materialProvider, paramList);
		pp.StartContext(context);
		Assert::AreEqual(0, (int)context.EventSummary().GetTotalEventCount());
		Assert::AreEqual(0, elementProvider.GetFailedElementBuildQueryCount());
		Assert::AreEqual(1, elementProvider.GetSuccessfulElementBuildQueryCount());
		context.ClearEventStatistics();
		elementProvider.ResetCounters();
		paramList.Destroy();
	}

	TEST_METHOD(TestBogusLineParse)
	{
		// our string to parse
		String line = _T("BOGUS LINE");

		// build our mock objects
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build our parameter list
		ParameterList& paramList = ParameterList::Create();
		paramList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("MY_PROP"), *new NullValue()));
		paramList.AddParameter(_T("PROPERTIES"), *properties);

		// prepare parser
		MockedPartParser parser(parentProvider, elementProvider, materialProvider, paramList);
		parser.StartContext(context);
		InputIterator begin = IteratorFactory::CreateStringIterator(line.begin());
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());

		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
	}

	TEST_METHOD(TestMistypedLineParse)
	{
		// our string to parse
		String line = _T("SET my_eset IS TEST_MAT WITH ()");

		// build our mock objects
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build our parameter list
		ParameterList& paramList = ParameterList::Create();
		paramList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("MY_PROP"), *new NullValue()));
		paramList.AddParameter(_T("PROPERTIES"), *properties);

		// prepare parser
		MockedPartParser parser(parentProvider, elementProvider, materialProvider, paramList);
		parser.StartContext(context);
		paramList.Destroy();
		InputIterator begin = IteratorFactory::CreateStringIterator(line.begin());
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());

		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
	}

	TEST_METHOD(TestInvalidMaterialTypeParse)
	{
		// our string to parse
		String line = _T("SET my_eset IS BOGUS_MAT WITH TEST_PROP = 1");

		// our analysis object
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		// build our mock objects.
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build our parameter list
		ParameterList& paramList = ParameterList::Create();
		paramList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("MY_PROP"), *new NullValue()));
		paramList.AddParameter(_T("PROPERTIES"), *properties);

		// prepare parser
		MockedPartParser parser(parentProvider, elementProvider, materialProvider, paramList);
		parser.StartContext(context);
		parser.SetAnalysis(ws);
		paramList.Destroy();
		InputIterator begin = IteratorFactory::CreateStringIterator(line.begin());
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());

		ParseResult result = parser.Parse(begin, end);

		// match should be ok, even though we couldn't accomplish to what is said
		Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

		// check if material provider was queried and we got the expected results
		Assert::AreEqual(1, materialProvider.GetIncorrectMaterialQueryCount());
		Assert::AreEqual(0, materialProvider.GetInvalidParamsQueryCount());
		Assert::AreEqual(0, materialProvider.GetSuccessfulQueryCount());

		// assert that the element set was not created
		Assert::AreEqual(false, analysis.ExistsElementSet(_T("my_eset")));

		// assert that the section definition for the element set was not created
		Assert::AreEqual(false, context.Sketches().HasSectionDefined(_T("my_eset")));

		ws.Destroy();
	}

	TEST_METHOD(TestInvalidMaterialParamsParse)
	{
		// our string to parse
		String line = _T("SET my_eset IS TEST_MAT WITH BOGUS_PROP");

		// our analysis object
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		// build our mock objects.
		ParseContextConcrete context;
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build our parameter list
		ParameterList& paramList = ParameterList::Create();
		paramList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("MY_PROP"), *new NullValue()));
		paramList.AddParameter(_T("PROPERTIES"), *properties);

		// prepare parser
		MockedPartParser parser(parentProvider, elementProvider, materialProvider, paramList);
		parser.StartContext(context);
		parser.SetAnalysis(ws);
		paramList.Destroy();
		InputIterator begin = IteratorFactory::CreateStringIterator(line.begin());
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());

		ParseResult result = parser.Parse(begin, end);

		// match should be ok, even though we couldn't accomplish to what is said
		Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

		// check if material provider was queried and we got the expected results
		Assert::AreEqual(0, materialProvider.GetIncorrectMaterialQueryCount());
		Assert::AreEqual(1, materialProvider.GetInvalidParamsQueryCount());
		Assert::AreEqual(0, materialProvider.GetSuccessfulQueryCount());

		// assert that the element set was not created
		Assert::AreEqual(false, analysis.ExistsElementSet(_T("my_eset")));

		// assert that the section definition for the element set was not created
		Assert::AreEqual(false, context.Sketches().HasSectionDefined(_T("my_eset")));

		ws.Destroy();
	}

	TEST_METHOD(TestSuccessfulParse)
	{
		// our string to parse
		String line = _T("SET my_eset IS TEST_MAT WITH TEST_PROP=1");

		// our analysis object
		NumericalModel& analysis = NumericalModel::Create();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		// build our mock objects.
		ParseContextConcrete context;
    SymbolTable& st = context.Symbols();
		MockProvider parentProvider;
		MockElementProvider elementProvider;
		MockMaterialProvider materialProvider;

		// build our parameter list
		ParameterList& paramList = ParameterList::Create();
		paramList.AddParameter(_T("ELEM_TYPE"), *new IdValue(_T("TEST_ELEMENT")));
		ArrayValue *properties = new ArrayValue();
		properties->AddValue(*new ParameterAssignment(_T("MY_PROP"), *new NullValue()));
		paramList.AddParameter(_T("PROPERTIES"), *properties);

		// prepare parser
		MockedPartParser parser(parentProvider, elementProvider, materialProvider, paramList);
		parser.StartContext(context);
		parser.SetAnalysis(ws);
		paramList.Destroy();
		InputIterator begin = IteratorFactory::CreateStringIterator(line.begin());
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());

		ParseResult result = parser.Parse(begin, end);

		// match should be ok, even though we couldn't accomplish to what is said
		Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

		// check if material provider was queried and we got the expected results
		Assert::AreEqual(0, materialProvider.GetIncorrectMaterialQueryCount());
		Assert::AreEqual(0, materialProvider.GetInvalidParamsQueryCount());
		Assert::AreEqual(1, materialProvider.GetSuccessfulQueryCount());

		// assert that the element set was created
		Assert::AreEqual(true, analysis.ExistsElementSet(_T("my_eset")));

		// test if a symbol was defined in this context
		Assert::AreEqual(true, st.IsSymbolDefined(_T("my_eset"), SymbolTable::kElementSet));

		// assert that the section definition for the element set was created
		Assert::AreEqual(true, context.Sketches().HasSectionDefined(_T("my_eset")));

		// test if a symbol was defined in this context
		Assert::AreEqual(true, st.IsSymbolDefined(_T("my_eset"), SymbolTable::kSection));

		// check if we have a well-formed section definition
		const SectionDefinition& section = context.Sketches().GetSection(_T("my_eset"));
		Assert::AreEqual(_T("TEST_ELEMENT"), section.GetSectionTypeName());
		Assert::AreEqual(true, section.IsPropertyDefined(_T("MY_PROP")));
		Assert::AreEqual(true, section.GetPropertyValue(_T("MY_PROP")).IsNull());

		// check if we have the expected material
		const TestMaterial& material = (const TestMaterial&)section.GetMaterial();
		Assert::AreEqual(true, material.IsTestMaterial());

		ws.Destroy();
	}

};

} } }

// include template method definitions due to compiler deficiency
#include "application/parsing/parsers/PartParser.cpp"

#endif


