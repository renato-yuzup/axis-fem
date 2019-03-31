#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "MockParseContext.hpp"
#include "MockClockwork.hpp"
#include "MockSolver.hpp"
#include "MockResultBucket.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/parsing/parsers/SnapshotParser.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/output/ResultBucket.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"

namespace af = axis::foundation;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aaj = axis::application::jobs;
namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace aao = axis::application::output;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asli = axis::services::language::iterators;
namespace aslf = axis::services::language::factories;
namespace aslp = axis::services::language::parsing;
namespace auc = axis::unit_tests::physalis;

// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const aslp::ParseResult::Result& v)
{
	return axis::String::int_parse((long)v).data();
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const asli::InputIterator& it)
{
	return std::wstring(1, *it);
}




namespace axis { namespace unit_tests { namespace physalis {


TEST_CLASS(SnapshotParserTest)
{
private:
	axis::application::jobs::StructuralAnalysis& CreateTestBenchWorkspace(void) const
	{
		aaj::StructuralAnalysis& ws = *new aaj::StructuralAnalysis(_T("."));

		// create two steps
		MockSolver   *m1 = new MockSolver(*new auc::MockClockwork(1));

		aaj::AnalysisStep &s1 = aaj::AnalysisStep::Create(0, 5, *m1, *new MockResultBucket());
		MockSolver   *m2 = new MockSolver(*new auc::MockClockwork(1));
		aaj::AnalysisStep &s2 = aaj::AnalysisStep::Create(5, 10, *m1, *new MockResultBucket());

		ws.AddStep(s1);
		ws.AddStep(s2);

		return ws;
	}
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
		aapps::SnapshotParser *sp = new aapps::SnapshotParser(false);
		delete sp;
	}

	TEST_METHOD(TestParseContext)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
		ipc.SetStepOnFocus(&ws.GetStep(0));
    sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// close context
		sp.CloseContext();
		ws.Destroy();
	}

	TEST_METHOD(TestParserBehaviorOnSuccess)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// close context
		sp.CloseContext();

		// no changes should have been made
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(0L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		ws.Destroy();
	}

	TEST_METHOD(TestDenyNestedBlocks)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// try to obtain any context
		try
		{
			sp.GetNestedContext(_T(""), aslse::ParameterList::Empty);
			Assert::Fail(_T("Expected exception not thrown."));
		}
		catch (axis::foundation::NotSupportedException&)
		{
			// ok, it was expected
		}
		catch (...)
		{	// what..?
			Assert::Fail(_T("Exception thrown, but it was not of the type expected."));
		}

		// close context
		sp.CloseContext();

		// no changes should have been made
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(0L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestAbsoluteStatement)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		axis::String testLine = _T("SNAPSHOT AT 2.1");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// mark should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());


		// close context
		sp.CloseContext();

		// check results
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(1L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		ada::SnapshotMark& sm = t1.GetSnapshotMark(0);
		Assert::AreEqual((real)2.1, sm.GetTime(), REAL_TOLERANCE);

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestRelativeStatement)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// test line 1
		axis::String testLine = _T("SNAPSHOT AGAIN AFTER 2");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// test line 2
		testLine = _T("SNAPSHOT AGAIN AFTER 1.2");
		begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
		end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// check results
    ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
    ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(2L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		ada::SnapshotMark& sm1 = t1.GetSnapshotMark(0);
		Assert::AreEqual((real)2.0, sm1.GetTime());
		ada::SnapshotMark& sm2 = t1.GetSnapshotMark(1);
		Assert::AreEqual((real)3.2, sm2.GetTime());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestTimeRangeStatement)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// test line 1
		axis::String testLine = _T("SNAPSHOT EVERY 0.4 FROM 0 TO 1.2");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// test line 2
		testLine = _T("SNAPSHOT EVERY 0.5 FROM 1.2 TO 2");
		begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
		end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// check results
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(5L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		ada::SnapshotMark& sm1 = t1.GetSnapshotMark(0);
		Assert::AreEqual((real)0.4, sm1.GetTime());
		ada::SnapshotMark& sm2 = t1.GetSnapshotMark(1);
		Assert::AreEqual((real)0.8, sm2.GetTime());
		ada::SnapshotMark& sm3 = t1.GetSnapshotMark(2);
		Assert::AreEqual((real)1.2, sm3.GetTime());
		ada::SnapshotMark& sm4 = t1.GetSnapshotMark(3);
		Assert::AreEqual((real)1.7, sm4.GetTime());
		ada::SnapshotMark& sm5 = t1.GetSnapshotMark(4);
		Assert::AreEqual((real)2.0, sm5.GetTime());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestSplitStatement)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		axis::String testLine = _T("DO 4 SNAPSHOTS");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// check results
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(4L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		ada::SnapshotMark& sm1 = t1.GetSnapshotMark(0);
		Assert::AreEqual((real)1.25, sm1.GetTime());
		ada::SnapshotMark& sm2 = t1.GetSnapshotMark(1);
		Assert::AreEqual((real)2.5, sm2.GetTime());
		ada::SnapshotMark& sm3 = t1.GetSnapshotMark(2);
		Assert::AreEqual((real)3.75, sm3.GetTime());
		ada::SnapshotMark& sm4 = t1.GetSnapshotMark(3);
		Assert::AreEqual((real)5.0, sm4.GetTime());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestUnexpectedStatement)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		axis::String testLine = _T("BOGUS LINE");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(false, result.IsMatch());
		Assert::AreEqual(aslp::ParseResult::FailedMatch, result.GetResult());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// check results
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(0L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestParserBehaviorOnParseError)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// test line 1
		axis::String testLine = _T("SNAPSHOT EVERY 0.4 FROM 0 TO 1.2");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// test line 2
		testLine = _T("BOGUS LINE");
    begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		result = sp.Parse(begin, end);
		Assert::AreEqual(false, result.IsMatch());
		Assert::AreEqual(aslp::ParseResult::FailedMatch, result.GetResult());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// because an error occurred, no marks should be placed
    ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
    ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(0L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestParserBehaviorOnMarkOrderError)
	{
		aapps::SnapshotParser sp(false);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// test line 1
		axis::String testLine = _T("SNAPSHOT EVERY 0.4 FROM 0 TO 1.2");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// test line 2
		testLine = _T("SNAPSHOT AT 0.9");
		begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
		end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// because an mark statements overlap, no marks should be
		// placed
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(0L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		Assert::AreEqual(1L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}

	TEST_METHOD(TestParserIgnoreStatementsBehavior)
	{
		aapps::SnapshotParser sp(true);
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		auc::MockParseContext ipc;

		// open context
    ipc.SetStepOnFocus(&ws.GetStep(0));
		sp.SetAnalysis(ws);
		sp.StartContext(ipc);

		// test line 1
		axis::String testLine = _T("SNAPSHOT EVERY 0.4 FROM 0 TO 1.2");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
		aslp::ParseResult result = sp.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// marks should be not be placed until close context
		Assert::AreEqual(0L, ws.GetStep(0).GetTimeline().SnapshotMarkCount());

		// close context
		sp.CloseContext();

		// due to ignore flag, no marks should be placed
		ada::AnalysisTimeline& t1 = ws.GetStep(0).GetTimeline();
		ada::AnalysisTimeline& t2 = ws.GetStep(1).GetTimeline();
		Assert::AreEqual(0L, t1.SnapshotMarkCount());
		Assert::AreEqual(0L, t2.SnapshotMarkCount());

		Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

		ws.Destroy();
	}
};


} } }

#endif

