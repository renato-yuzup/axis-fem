#ifndef __AXIS_MODULES_HPP
#define __AXIS_MODULES_HPP


/* Provedores de parsers */
#define AXIS_PROVIDER_PARSER_ANALYSIS						"axis.base.providers.input.AnalysisProvider"
#define AXIS_PROVIDER_PARSER_NODES							"axis.base.providers.input.NodeProvider"
#define AXIS_PROVIDER_PARSER_NODESET						"axis.base.providers.input.NodeSetProvider"
#define AXIS_PROVIDER_ELEMENT								"axis.base.providers.input.FiniteElementProvider"
#define AXIS_PROVIDER_CONSTITUTIVE_MODEL					"axis.base.providers.input.ConstitutiveModelProvider"

#define AXIS_PROVIDER_GAUSSPOINT							"axis.base.providers.GaussPointProvider"
#define AXIS_PROVIDER_SHAPEFUNCTION							"axis.base.providers.ShapeFunctionProvider"

#define AXIS_FACTORIES_NODE									"axis.base.factories.Node"

#define AXIS_SHAPEFUNCTION_LINEARTETRAHEDRON				"axis.base.shapeFunctions.LinearTetrahedron"
#define AXIS_SHAPEFUNCTION_LINEARHEXAHEDRON					"axis.base.shapeFunctions.LinearHexahedron"


#define AXIS_STORE_ELEMENTS									"axis.base.stores.ElementStore"
#define AXIS_STORE_NODES									"axis.base.stores.NodeStore"
#define AXIS_STORE_LOADS									"axis.base.stores.LoadStores"
#define AXIS_STORE_CONSTRAINTS								"axis.base.stores.ContraintStore"

#endif
