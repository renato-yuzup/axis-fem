#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "MockParseContext.hpp"
#include "MockSolver.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/parsing/parsers/SnapshotParser.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "application/parsing/parsers/StepParser.hpp"
#include "application/factories/parsers/StepParserProvider.hpp"
#include "application/locators/CollectorFactoryLocator.hpp"
#include "application/locators/WorkbookFactoryLocator.hpp"
#include "MockWorkbookFactory.hpp"
#include "MockSolverFactory.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/StringValue.hpp"

namespace aal = axis::application::locators;
namespace af = axis::foundation;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aaj = axis::application::jobs;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asli = axis::services::language::iterators;
namespace aslf = axis::services::language::factories;
namespace aslp = axis::services::language::parsing;
namespace auc = axis::unit_tests::physalis;

// HACK: Yeah, it is ugly, but necessary...
#include "application/parsing/parsers/StepParser.cpp" // needed for template instantiation

namespace axis { namespace unit_tests { namespace physalis {

TEST_CLASS(StepParserTest)
{
private:
  typedef aapps::StepParserTemplate<aal::SolverFactoryLocator, aal::ClockworkFactoryLocator> 
          MockStepParser;

  aal::SolverFactoryLocator _mockLocator;
  axis::application::jobs::StructuralAnalysis& CreateTestBenchStructuralAnalysis(void) const
  {
    return *new aaj::StructuralAnalysis(_T("."));
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

  TEST_METHOD(TestConstructorWithValidParameters)
  {

    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("MOCK_SOLVER"), 0, 1, 
                                            aslse::ParameterList::Empty, cfl, wfl);
    delete sp;
  }

  TEST_METHOD(TestConstructorWithInvalidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = NULL;
    try
    {
      sp = new MockStepParser(spp, _T(""), _mockLocator, _T("BOGUS_SOLVER"), 0, 1, 
                              aslse::ParameterList::Empty, cfl, wfl);
    }
    catch (...)
    {	// no exception should have been thrown, even though parameters are invalid (postpone exceptions)
      Assert::Fail(_T("It was expected that exceptions would be postponed to StartContext()"));
    }

    delete sp;
  }

  TEST_METHOD(TestStartContextOnValidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("MOCK_SOLVER"), 0, 1, 
                                            aslse::ParameterList::Empty, cfl, wfl);
    MockSolverFactory msf;
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    _mockLocator.RegisterFactory(msf);
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);

    // should have created an analysis step
    Assert::AreEqual(1, ws.GetStepCount());
    Assert::AreEqual(0, ws.GetStep(0).GetStartTime(), REAL_TOLERANCE);
    Assert::AreEqual(1, ws.GetStep(0).GetEndTime(), REAL_TOLERANCE);

    sp->CloseContext();
    Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

    delete sp;
    ws.Destroy();
    _mockLocator.UnregisterFactory(msf);
  }

  TEST_METHOD(TestStartContextOnInvalidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("BOGUS_SOLVER"), 0, 1, 
                                            aslse::ParameterList::Empty, cfl, wfl);
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);

    Assert::AreEqual(0, ws.GetStepCount());
    Assert::AreEqual(1L, (long)ipc.EventSummary().GetTotalEventCount());

    sp->CloseContext();
    delete sp;
    ws.Destroy();
  }

  TEST_METHOD(TestSnapshotBlockOnValidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("MOCK_SOLVER"), 0, 1, 
                                            aslse::ParameterList::Empty, cfl, wfl);
    MockSolverFactory msf;
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    _mockLocator.RegisterFactory(msf);
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);

    aapps::BlockParser *bp;
    try
    {
      bp = &sp->GetNestedContext(_T("SNAPSHOTS"), aslse::ParameterList::Empty);		
      delete bp;
    }
    catch (axis::foundation::NotSupportedException&)
    {	// no, it should have accepted. Why...?
      Assert::Fail(_T("The SNAPSHOTS block was not recognized. Why...?"));
    }
    catch (...)
    {
      Assert::Fail(_T("The parser threw an exception not defined in our standards."));
    }

    Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());

    sp->CloseContext();
    delete sp;
    ws.Destroy();
    _mockLocator.UnregisterFactory(msf);
  }

  TEST_METHOD(TestSnapshotBlockOnInvalidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("BOGUS_SOLVER"), 0, 1, 
                                            aslse::ParameterList::Empty, cfl, wfl);
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);
    Assert::AreEqual(1L, (long)ipc.EventSummary().GetTotalEventCount());

    aapps::BlockParser *bp;
    try
    {
      bp = &sp->GetNestedContext(_T("SNAPSHOTS"), aslse::ParameterList::Empty);		
      delete bp;
      Assert::Fail(_T("The SNAPSHOTS block was recognized on error condition."));
    }
    catch (axis::foundation::NotSupportedException&)
    {	// ok, this was expected
    }
    catch (...)
    {
      Assert::Fail(_T("The parser threw an exception not defined in our standards."));
    }

    Assert::AreEqual(1L, (long)ipc.EventSummary().GetTotalEventCount());

    sp->CloseContext();
    delete sp;
    ws.Destroy();
  }

  TEST_METHOD(TestRefuteStatements)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("MOCK_SOLVER"), 0, 1, 
                                            aslse::ParameterList::Empty, cfl, wfl);
    MockSolverFactory msf;
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    _mockLocator.RegisterFactory(msf);
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);

    String testLine = _T("BOGUS LINE");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(testLine.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(testLine.end());
    aslp::ParseResult result = sp->Parse(begin, end);
    Assert::AreEqual(true, result.IsMatch());


    Assert::AreEqual(1L, (long)ipc.EventSummary().GetTotalEventCount());

    sp->CloseContext();

    delete sp;
    ws.Destroy();
    _mockLocator.UnregisterFactory(msf);
  }

  TEST_METHOD(TestResultBlockOnValidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockWorkbookFactory &mwf = *new MockWorkbookFactory();
    MockSolverFactory &msf = *new MockSolverFactory();
    wfl.RegisterFactory(mwf);
    _mockLocator.RegisterFactory(msf);
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("MOCK_SOLVER"), 0, 1, 
      aslse::ParameterList::Empty, cfl, wfl);
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);

    aapps::BlockParser *bp;
    aslse::ParameterList& params = aslse::ParameterList::Create();
    params.AddParameter(_T("FORMAT"), *new aslse::IdValue(_T("TEST_FORMAT")));
    params.AddParameter(_T("FILE"), *new aslse::StringValue(_T("myfile")));
    try
    {
      bp = &sp->GetNestedContext(_T("OUTPUT"), params);		
      params.Destroy();
    }
    catch (axis::foundation::NotSupportedException&)
    {	// no, it should have accepted. Why...?
      params.Destroy();
      Assert::Fail(_T("The OUTPUT block was not recognized. Why...?"));
    }
    catch (...)
    {
      params.Destroy();
      Assert::Fail(_T("The parser threw an exception not defined in our standards."));
    }

    delete bp;
    Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());
    sp->CloseContext();
    delete sp;
    ws.Destroy();
  }

  TEST_METHOD(TestResultBlockOnInvalidParameters)
  {
    aafp::StepParserProvider spp;
    aal::CollectorFactoryLocator cfl;
    aal::WorkbookFactoryLocator wfl;
    MockWorkbookFactory &mwf = *new MockWorkbookFactory();
    MockSolverFactory &msf = *new MockSolverFactory();
    wfl.RegisterFactory(mwf);
    _mockLocator.RegisterFactory(msf);
    MockStepParser *sp = new MockStepParser(spp, _T(""), _mockLocator, _T("MOCK_SOLVER"), 0, 1, 
      aslse::ParameterList::Empty, cfl, wfl);
    aaj::StructuralAnalysis& ws = CreateTestBenchStructuralAnalysis();
    auc::MockParseContext ipc;
    sp->SetAnalysis(ws);
    sp->StartContext(ipc);

    aapps::BlockParser *bp;
    aslse::ParameterList& params = aslse::ParameterList::Create();
    params.AddParameter(_T("BOGUS_FORMAT"), *new aslse::IdValue(_T("BOGUS_FORMAT")));
    params.AddParameter(_T("FILE"), *new aslse::StringValue(_T("myfile")));
    try
    {
      bp = &sp->GetNestedContext(_T("OUTPUT"), params);		
      delete bp;
      params.Destroy();
      Assert::Fail(_T("The bogus OUTPUT block was accepted. Why...?"));
    }
    catch (axis::foundation::NotSupportedException&)
    {	// that's fine
      params.Destroy();
    }
    catch (...)
    {
      params.Destroy();
      Assert::Fail(_T("The parser threw an exception not defined in our standards."));
    }

    Assert::AreEqual(0L, (long)ipc.EventSummary().GetTotalEventCount());
    sp->CloseContext();
    delete sp;
    ws.Destroy();
  }

};


} } } 

#endif

