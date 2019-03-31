#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"
#include "System.hpp"
#include "AxisString.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "MockSolverFactoryLocator.hpp"
#include "application/factories/parsers/StepParserProvider.hpp"
#include "services/management/GlobalProviderCatalogImpl.hpp"
#include "MockSolverFactoryLocator.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/locators/ClockworkFactoryLocator.hpp"
#include "Application/Locators/WorkbookFactoryLocator.hpp"
#include "Application/Locators/CollectorFactoryLocator.hpp"


using namespace axis::foundation;
using namespace axis::application::factories::parsers;
using namespace axis::application::locators;
using namespace axis::application::parsing::parsers;
using namespace axis::services::management;
using namespace axis::services::language::syntax::evaluation;

namespace aal = axis::application::locators;



namespace axis { namespace unit_tests { namespace core {

TEST_CLASS(StepParserProviderTest)
{
private:
	MockSolverFactoryLocator _mockLocator;
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
		StepParserProvider *spp = new StepParserProvider();
		delete spp;
	}

	TEST_METHOD(TestProviderBehaviorOnRegistering)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		delete &manager;
	}

	TEST_METHOD(TestCanParseWithValidParameters)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("MOCK_SOLVER")));
		params.AddParameter(_T("START_TIME"), *new NumberValue(0L));
		params.AddParameter(_T("END_TIME"), *new NumberValue(1L));

		bool ok = spp.CanParse(_T("STEP"), params);
		params.Destroy();

		Assert::AreEqual(true, ok);

		delete &manager;
	}

	TEST_METHOD(TestCanParseWithInvalidParameters)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("MOCK_SOLVER")));
		params.AddParameter(_T("START_TIME"), *new IdValue(_T("BOGUS")));	// incorrect
		params.AddParameter(_T("END_TIME"), *new NumberValue(1L));

		bool ok = spp.CanParse(_T("STEP"), params);
		params.Destroy();

		Assert::AreEqual(false, ok);

		delete &manager;
	}

	TEST_METHOD(TestCanParseWithMissingParameters)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("MOCK_SOLVER")));
		params.AddParameter(_T("START_TIME"), *new NumberValue(0L));	// END_TIME is missing

		bool ok = spp.CanParse(_T("STEP"), params);
		params.Destroy();

		Assert::AreEqual(false, ok);

		delete &manager;
	}

	TEST_METHOD(TestCanParseWithUnrecognizedParameters)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("MOCK_SOLVER")));
		params.AddParameter(_T("START_TIMES"), *new NumberValue(0L));	// note incorrect spelling
		params.AddParameter(_T("END_TIME"), *new NumberValue(1L));

		bool ok = spp.CanParse(_T("STEP"), params);
		params.Destroy();

		Assert::AreEqual(false, ok);

		delete &manager;
	}

	TEST_METHOD(TestCanParseWithParametersInExcess)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("MOCK_SOLVER")));
		params.AddParameter(_T("START_TIME"), *new NumberValue(0L));	
		params.AddParameter(_T("END_TIME"), *new NumberValue(1L));
		params.AddParameter(_T("MY_BOGUS_NUMBER"), *new NumberValue(2L));	// unnecessary parameter

		bool ok = spp.CanParse(_T("STEP"), params);
		params.Destroy();

		Assert::AreEqual(true, ok);	// it still accepts, because it's up to the 
		// step parser refuse the dangling parameter

		delete &manager;
	}

	TEST_METHOD(TestCanParseWithUnknownSolverType)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
    aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

    manager.RegisterProvider(sfl);
    manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
    manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("BOGUS_SOLVER")));	// inexistent solver
		params.AddParameter(_T("START_TIME"), *new NumberValue(0L));	
		params.AddParameter(_T("END_TIME"), *new NumberValue(1L));

		bool ok = spp.CanParse(_T("STEP"), params);
		params.Destroy();

		Assert::AreEqual(true, ok);	// yes, even though it can't build the solver, we can still
		// accept the context (though we will throw off its entire
		// contents)

		delete &manager;
	}

	TEST_METHOD(TestBuildParser)
	{
		GlobalProviderCatalogImpl& manager = *new GlobalProviderCatalogImpl();
		MockSolverFactoryLocator sfl;
		StepParserProvider spp;
		aal::ClockworkFactoryLocator cff;
    aal::WorkbookFactoryLocator wfl;
    aal::CollectorFactoryLocator cfl;

		manager.RegisterProvider(sfl);
		manager.RegisterProvider(cff);
    manager.RegisterProvider(wfl);
    manager.RegisterProvider(cfl);
		manager.RegisterProvider(spp);

		ParameterList& params = ParameterList::Create();
		params.AddParameter(_T("TYPE"), *new IdValue(_T("MOCK_SOLVER")));	// inexistent solver
		params.AddParameter(_T("START_TIME"), *new NumberValue(0L));	
		params.AddParameter(_T("END_TIME"), *new NumberValue(1L));

		BlockParser& parser = spp.BuildParser(_T("STEP"), params);
		params.Destroy();

		delete &parser;
		delete &manager;
	}


};



} } }

#endif

