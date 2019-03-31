#include "CorePluginAgent.hpp"

// Locator includes
#include "application/locators/ConstraintParserLocator.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "application/locators/WorkbookFactoryLocator.hpp"
#include "application/locators/MaterialFactoryLocator.hpp"
#include "application/locators/SolverFactoryLocator.hpp"
#include "application/locators/CollectorFactoryLocator.hpp"
#include "application/locators/ClockworkFactoryLocator.hpp"

// Parser factory includes
#include "application/factories/parsers/RootParserProvider.hpp"
#include "application/factories/parsers/CurveScopeProvider.hpp"
#include "application/factories/parsers/LoadScopeProvider.hpp"
#include "application/factories/parsers/StepParserProvider.hpp"
#include "application/factories/parsers/AnalysisBlockParserProvider.hpp"
#include "application/factories/parsers/NodeSetParserProvider.hpp"
#include "application/factories/parsers/ElementSetParserProvider.hpp"
#include "application/factories/parsers/PartParserProvider.hpp"
#include "application/factories/parsers/NodeParserProvider.hpp"

// Solver and clockwork includes
// #include "application/factories/algorithms/ExplicitStandardTimeSolverFactory.hpp"
// #include "application/factories/algorithms/RegularClockworkFactory.hpp"
// #include "application/factories/algorithms/WaveSpeedProportionalClockworkFactory.hpp"
// #include "application/factories/algorithms/LinearStaticCGSolverFactory.hpp"

// Other factories
#include "application/factories/elements/NodeFactory.hpp"

// Auxiliary classes
#include "services/management/ServiceLocator.hpp"

namespace aaa = axis::application::agents;
namespace asmg = axis::services::management;

namespace aal = axis::application::locators;
namespace aafa = axis::application::factories::algorithms;
namespace aafbc = axis::application::factories::boundary_conditions;
namespace aafc = axis::application::factories::collectors;
namespace aafe = axis::application::factories::elements;
namespace aafw = axis::application::factories::workbooks;
namespace aafp = axis::application::factories::parsers;

namespace asmg = axis::services::management;

aaa::CorePluginAgent::CorePluginAgent( void )
{
	// nothing to do here
}

aaa::CorePluginAgent::~CorePluginAgent( void )
{
	// nothing to do here
}

void aaa::CorePluginAgent::SetUp( asmg::GlobalProviderCatalog& manager )
{
	_manager = &manager;
}


void aaa::CorePluginAgent::RegisterCoreProviders( void )
{
	RegisterFeatureLocators(*_manager);
	RegisterInputProviders(*_manager);
	RegisterSystemFactories(*_manager);
}

void aaa::CorePluginAgent::RegisterFeatureLocators( asmg::GlobalProviderCatalog& manager )
{
	// these are the locators that we will register
	aal::ConstraintParserLocator& constraintLocator	  = *new aal::ConstraintParserLocator();
	aal::ElementParserLocator& elementLocator				  = *new aal::ElementParserLocator();
	aal::WorkbookFactoryLocator& formatLocator			  = *new aal::WorkbookFactoryLocator();
	aal::MaterialFactoryLocator& materialLocator			= *new aal::MaterialFactoryLocator();
	aal::SolverFactoryLocator& solverLocator			    = *new aal::SolverFactoryLocator();
	aal::ClockworkFactoryLocator& clockworkLocator		= *new aal::ClockworkFactoryLocator();
	aal::CollectorFactoryLocator& collectorLocator	  = *new aal::CollectorFactoryLocator();

	manager.RegisterProvider(constraintLocator);
	manager.RegisterProvider(elementLocator);
	manager.RegisterProvider(formatLocator);
	manager.RegisterProvider(materialLocator);
	manager.RegisterProvider(solverLocator);
	manager.RegisterProvider(clockworkLocator);
	manager.RegisterProvider(collectorLocator);
}

void aaa::CorePluginAgent::RegisterInputProviders( asmg::GlobalProviderCatalog& manager )
{
	// get constraint and collector locators (we will need them
	// when nesting contexts)
	aal::ConstraintParserLocator& constraintLocator = static_cast<aal::ConstraintParserLocator&>(
			manager.GetProvider(asmg::ServiceLocator::GetNodalConstraintParserLocatorPath()));
	aal::ElementParserLocator& elementLocator = static_cast<aal::ElementParserLocator&>(
			manager.GetProvider(asmg::ServiceLocator::GetFiniteElementLocatorPath()));

	// instantiate block providers
	aafp::BlockProvider& analysisProvider						= *new aafp::RootParserProvider();
	aafp::BlockProvider& curveScopeProvider					= *new aafp::CurveScopeProvider();
	aafp::BlockProvider& loadScopeProvider					= *new aafp::LoadScopeProvider();
	aafp::BlockProvider& stepProvider							  = *new aafp::StepParserProvider();
	aafp::BlockProvider& analysisSettingsProvider		= *new aafp::AnalysisBlockParserProvider();
	aafp::BlockProvider& nodeSetProvider						= *new aafp::NodeSetParserProvider();
	aafp::BlockProvider& elementSetProvider					= *new aafp::ElementSetParserProvider();
	aafp::BlockProvider& partProvider							  = *new aafp::PartParserProvider();

	manager.RegisterProvider(analysisProvider);
	manager.RegisterProvider(curveScopeProvider);
	manager.RegisterProvider(loadScopeProvider);
	manager.RegisterProvider(stepProvider);
	manager.RegisterProvider(analysisSettingsProvider);
	manager.RegisterProvider(nodeSetProvider);
	manager.RegisterProvider(elementSetProvider);
	manager.RegisterProvider(partProvider);		// must be registered after element and material provider!

	// nest input providers -- very important if we want to parse blocks 
	// and sub-blocks correctly :)
	// these goes right under the root of the input file
	analysisProvider.RegisterProvider(curveScopeProvider);
	analysisProvider.RegisterProvider(analysisSettingsProvider);
	analysisProvider.RegisterProvider(elementSetProvider);
	analysisProvider.RegisterProvider(nodeSetProvider);
	analysisProvider.RegisterProvider(partProvider);
	analysisProvider.RegisterProvider(elementLocator);

	// these goes under the analysis settings block
	analysisSettingsProvider.RegisterProvider(stepProvider);

	// these goes under the step block 
	stepProvider.RegisterProvider(loadScopeProvider);
	stepProvider.RegisterProvider(constraintLocator);
}

void aaa::CorePluginAgent::RegisterSystemFactories( asmg::GlobalProviderCatalog& manager )
{
	aal::ConstraintParserLocator& constraintLocator  = static_cast<aal::ConstraintParserLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetNodalConstraintParserLocatorPath()));
	aal::CollectorFactoryLocator& collectorLocator   = static_cast<aal::CollectorFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetCollectorFactoryLocatorPath()));
	aal::SolverFactoryLocator& solverLocator         = static_cast<aal::SolverFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetSolverLocatorPath()));
	aal::ClockworkFactoryLocator& clockworkLocator   = static_cast<aal::ClockworkFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetClockworkFactoryLocatorPath()));
	aal::WorkbookFactoryLocator& formatLocator       = static_cast<aal::WorkbookFactoryLocator&>(
    manager.GetProvider(asmg::ServiceLocator::GetWorkbookFactoryLocatorPath()));
// 
//   // solver factories
// 	solverLocator.RegisterFactory(*new LinearStaticCGSolverFactory());			 // linear static solver
// 	solverLocator.RegisterFactory(*new ExplicitStandardTimeSolverFactory()); // standard explicit solver
// 
// 	// clockwork factories
// 	clockworkLocator.RegisterFactory(*new RegularClockworkFactory());
// 	clockworkLocator.RegisterFactory(*new WaveSpeedProportionalClockworkFactory());
}

void aaa::CorePluginAgent::RegisterSystemCustomizableProviders( void )
{
	// these providers are system-specific, but can be replaced by
	// custom providers
	aafe::NodeFactory& nodeFactory    = *new aafe::NodeFactory();
	aafp::BlockProvider& nodeProvider = *new aafp::NodeParserProvider();
	// register them
	_manager->RegisterProvider(nodeFactory);
	_manager->RegisterProvider(nodeProvider);
	// nest in the corresponding provider
	aafp::BlockProvider& analysisProvider = (aafp::BlockProvider&)
    _manager->GetProvider(asmg::ServiceLocator::GetMasterInputParserProviderPath());
	analysisProvider.RegisterProvider(nodeProvider);
}