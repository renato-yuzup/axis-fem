#include <tchar.h>
#include <cfixcc.h>
#include "AxisString.hpp"
#include "application/factories/materials/LinearIsoElasticFactory.hpp"
#include "application/locators/MaterialFactoryLocator.hpp"
#include "MockProviders.hpp"

using namespace axis::foundation;
using namespace axis::application::factories::Base;
using namespace axis::application::factories::materials;
using namespace axis::application::parsing::parsers::Base;
using namespace axis::domain::materials::Base;


/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
class MaterialFactoryTestFixture : public cfixcc::TestFixture
{
public:
	void TestLinearIsoElasticCanBuildMethodFail(void);
	void TestLinearIsoElasticCanBuildMethodPass(void);
	void TestLinearIsoElasticBuildMethod(void);
	void TestMaterialLocatorBuildMethod(void);
};

CFIXCC_BEGIN_CLASS( MaterialFactoryTestFixture )
	CFIXCC_METHOD( TestLinearIsoElasticCanBuildMethodFail )
	CFIXCC_METHOD( TestLinearIsoElasticCanBuildMethodPass )
	CFIXCC_METHOD( TestLinearIsoElasticBuildMethod )
	CFIXCC_METHOD( TestMaterialLocatorBuildMethod )
CFIXCC_END_CLASS()

