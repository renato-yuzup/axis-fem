#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "services/management/WindowsPluginConnector.hpp"
#include "foundation/ApplicationErrorException.hpp"
#include "services/management/GlobalProviderCatalogImpl.hpp"

using namespace axis::foundation;
using namespace axis::services::management;
using namespace axis::services::management;


namespace axis { namespace unit_tests { namespace core {

/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
TEST_CLASS(WindowsPluginLoaderTest)
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

	TEST_METHOD(TestCanLoadPlugin)
	{
		WindowsPluginConnector plugin(_T("axis.StandardLibrary.dll"));
		GlobalProviderCatalogImpl manager;
		try
		{
			plugin.LoadPlugin();
		}
		catch (axis::foundation::ApplicationErrorException)
		{	// this is bad...
			Assert::Fail(_T("Severe error condition happened!!"));
		}

		Assert::AreEqual(true, plugin.IsPluginLoaded());

		plugin.UnloadPlugin(manager);	// I know, this looks creepy
	}
};


} } }

#endif

