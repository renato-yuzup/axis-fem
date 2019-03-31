#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"
#include <math.h>

#include "System.hpp"
#include "AxisString.hpp"
#include "MockParseContext.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/parsers/MultiLineCurveParser.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "domain/curves/MultiLineCurve.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "foundation/NotSupportedException.hpp"

using namespace axis;
using namespace axis::foundation;
using namespace axis::application::parsing::core;
using namespace axis::application::parsing::parsers;
using namespace axis::domain::materials;
using namespace axis::application::jobs;
using namespace axis::domain::analyses;
using namespace axis::domain::curves;
using namespace axis::services::language::iterators;
using namespace axis::services::language::factories;
using namespace axis::services::language::parsing;
namespace adcu = axis::domain::curves;
namespace afm = axis::foundation::memory;

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


namespace axis { namespace unit_tests { namespace orange {


TEST_CLASS(MultiLineCurveParserTest)
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

	TEST_METHOD(TestGoodScenario)
	{
		MultiLineCurveParser& parser = *new MultiLineCurveParser(_T("my_id"));
		NumericalModel& analysis = NumericalModel::Create();
		MockParseContext context;
    SymbolTable& st = context.Symbols();
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		parser.SetAnalysis(ws);
		parser.StartContext(context);

		String line = _T("(0, 10)");
		InputIterator begin = IteratorFactory::CreateStringIterator(line);
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		line = _T("<\t5.02     \t\t   -4.2e-6  \t>");
		begin = IteratorFactory::CreateStringIterator(line);
		end = IteratorFactory::CreateStringIterator(line.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		line = _T("7 20");
		begin = IteratorFactory::CreateStringIterator(line);
		end = IteratorFactory::CreateStringIterator(line.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		parser.CloseContext();

		// verify the curve was added correctly
		Assert::AreEqual(true, analysis.Curves().Contains(_T("my_id")));
		axis::domain::curves::MultiLineCurve& c = (axis::domain::curves::MultiLineCurve&)analysis.Curves()[_T("my_id")];

		// test if a symbol was defined in this context
		Assert::AreEqual(true, st.IsSymbolDefined(_T("my_id"), SymbolTable::kCurve));

		// test if curve parameters are correct
		Assert::AreEqual((size_t)3, c.PointCount());
		Assert::AreEqual((real)10.0, c.GetValueAt(0), REAL_TOLERANCE);
		Assert::AreEqual((real)-4.2e-6, c.GetValueAt((real)5.02), REAL_TOLERANCE);
		Assert::AreEqual((real)20.0, c.GetValueAt(7), REAL_TOLERANCE);

		ws.Destroy();
		delete &parser;
	}

	TEST_METHOD(TestBadSyntax)
	{
		MultiLineCurveParser& parser = *new MultiLineCurveParser(_T("my_id"));
		NumericalModel& analysis = NumericalModel::Create();
		MockParseContext context;
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		parser.SetAnalysis(ws);
		parser.StartContext(context);

		String line = _T("(0, 10)");
		InputIterator begin = IteratorFactory::CreateStringIterator(line);
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		// everything should go ok until here

		line = _T("<\t5.02     ::\t\t   -4.2e-6  \t>");
		begin = IteratorFactory::CreateStringIterator(line);
		end = IteratorFactory::CreateStringIterator(line.end());
		result = parser.Parse(begin, end);

		// wrong syntax
		Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
		Assert::AreNotEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());

		// should go fine with this one
		line = _T("7 20");
		begin = IteratorFactory::CreateStringIterator(line);
		end = IteratorFactory::CreateStringIterator(line.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());

		parser.CloseContext();

		// verify the curve was NOT added
		Assert::AreEqual(false, analysis.Curves().Contains(_T("my_id")));

		ws.Destroy();
		delete &parser;
	}

	TEST_METHOD(TestFailOnExistentCurve)
	{
		MultiLineCurveParser& parser = *new MultiLineCurveParser(_T("my_id"));
		NumericalModel& analysis = NumericalModel::Create();
		MockParseContext context;
		StructuralAnalysis& ws = *new StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		// add a mock curve under the same id
		afm::RelativePointer curvePtr = adcu::MultiLineCurve::Create(1);
		analysis.Curves().Add(_T("my_id"), curvePtr);

		parser.SetAnalysis(ws);
		parser.StartContext(context);

		String line = _T("(0, 10)");
		InputIterator begin = IteratorFactory::CreateStringIterator(line);
		InputIterator end = IteratorFactory::CreateStringIterator(line.end());
		ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(0, (int)context.EventSummary().GetTotalEventCount());

		line = _T("<\t5.02     \t\t   -4.2e-6  \t>");
		begin = IteratorFactory::CreateStringIterator(line);
		end = IteratorFactory::CreateStringIterator(line.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(0, (int)context.EventSummary().GetTotalEventCount());

		line = _T("7 20");
		begin = IteratorFactory::CreateStringIterator(line);
		end = IteratorFactory::CreateStringIterator(line.end());
		result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());
		Assert::AreEqual(0, (int)context.EventSummary().GetTotalEventCount());

		parser.CloseContext();

		// we must still have only one curve
		Assert::AreEqual((size_type)1, analysis.Curves().Count());

		ws.Destroy();
		delete &parser;
	}

	TEST_METHOD(TestFailOnSubBlocks)
	{
		MultiLineCurveParser& parser = *new MultiLineCurveParser(_T("my_id"));
		axis::services::language::syntax::evaluation::ParameterList& bogusParamList = 
      axis::services::language::syntax::evaluation::ParameterList::Create();

		// should trigger an exception
		try
		{
			parser.GetNestedContext(_T("BOGUS"), bogusParamList);

			// should not reach this line
			Assert::Fail(_T("Exception not thrown."));
		}
		catch (axis::foundation::NotSupportedException&)
		{
			// ok, what we are expecting
		}
		catch (...)
		{	// uh oh, wrong exception
			Assert::Fail(_T("Wrong exception thrown."));
		}

		delete &bogusParamList;
		delete &parser;
	}

};

} } }

#endif

