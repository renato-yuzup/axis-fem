#include "ServiceLocator.hpp"

namespace asmg = axis::services::management;

asmg::ServiceLocator::ServiceLocator( void )
{
	// nothing to do here
}

asmg::ServiceLocator::~ServiceLocator( void )
{
	// nothing to do here
}

const axis::AsciiString asmg::ServiceLocator::GetCurveProviderBasePath( void )
{
	return GetRootProviderBasePath() + ".curves";
}

const axis::AsciiString asmg::ServiceLocator::GetRootProviderBasePath( void )
{
	return "axis.base.providers.input";
}

const char * asmg::ServiceLocator::GetFiniteElementLocatorPath( void )
{
	return "axis.base.locators.finite_element_parser";
}

const char * asmg::ServiceLocator::GetMaterialFactoryLocatorPath( void )
{
	return "axis.base.locators.material_model_factory";
}

const char * asmg::ServiceLocator::GetNodalConstraintParserLocatorPath( void )
{
	return "axis.base.locators.constraint_parser";
}

const char * asmg::ServiceLocator::GetCollectorFactoryLocatorPath( void )
{
	return "axis.base.locators.result_collector_builder";
}

const char * asmg::ServiceLocator::GetPartInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.part_assembler";
}

const char * asmg::ServiceLocator::GetNodalLoadInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.nodal_load";
}

const char * asmg::ServiceLocator::GetLoadSectionInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.load_section";
}

const char * asmg::ServiceLocator::GetCurveSectionInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.curve_section";
}

const char * asmg::ServiceLocator::GetNodeSetInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.node_set";
}

const char * asmg::ServiceLocator::GetNodeInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.node";
}

const char * asmg::ServiceLocator::GetSolverLocatorPath( void )
{
	return "axis.base.locators.solver_factory";
}

const char * asmg::ServiceLocator::GetElementSetInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.element_set";
}

const char * asmg::ServiceLocator::GetNodeFactoryPath( void )
{
	return "axis.base.features.factories.mesh.node";
}

const char * asmg::ServiceLocator::GetMasterInputParserProviderPath( void )
{
	return "axis.base.features.providers.parsing.numerical_analysis";
}

const char * asmg::ServiceLocator::GetWorkbookFactoryLocatorPath( void )
{
	return "axis.base.locators.format_builder_locator_factory";
}

const char * asmg::ServiceLocator::GetClockworkFactoryLocatorPath( void )
{
	return "axis.base.locators.clockwork_factory";
}
