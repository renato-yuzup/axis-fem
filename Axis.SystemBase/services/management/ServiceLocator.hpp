#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			/**********************************************************************************************//**
			 * @brief	Provides information about known root providers.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	28 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API ServiceLocator
			{
			private:

				/**********************************************************************************************//**
				 * @brief	Private constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	28 ago 2012
				 **************************************************************************************************/
				ServiceLocator(void);
			public:

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	28 ago 2012
				 **************************************************************************************************/
				~ServiceLocator(void);

				/**********************************************************************************************//**
				 * @brief	Returns the base path of all parser providers.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	28 ago 2012
				 *
				 * @return	The provider base path.
				 **************************************************************************************************/
				static const axis::AsciiString GetRootProviderBasePath(void);

				/**********************************************************************************************//**
				 * @brief	Returns the base path for registered curve parser providers.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	28 ago 2012
				 *
				 * @return	The curve parser provider base path.
				 **************************************************************************************************/
				static const axis::AsciiString GetCurveProviderBasePath(void);
				

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered finite element formulations and types parsers.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetFiniteElementLocatorPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered material model parsers.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetMaterialFactoryLocatorPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered nodal constraint parsers.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetNodalConstraintParserLocatorPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered result collector builders.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetCollectorFactoryLocatorPath(void);

        /**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for part/assembly of elements.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetPartInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for nodal loads.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetNodalLoadInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for the loads section.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetLoadSectionInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for the curves section.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetCurveSectionInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for node sets.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetNodeSetInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for mesh nodes.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetNodeInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered solver factories.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetSolverLocatorPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered clockwork factories.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetClockworkFactoryLocatorPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the locator which knows
				 * 			 registered output format factories.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetWorkbookFactoryLocatorPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the input block parser for element sets.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetElementSetInputParserProviderPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard node
				 * 			 factory.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetNodeFactoryPath(void);

				/**********************************************************************************************//**
				 * <summary> Returns the plugin path for the standard provider for
				 * 			 the root input block parser.</summary>
				 *
				 * <returns> A pointer to a null-terminated string.</returns>
				 **************************************************************************************************/
				static const char * GetMasterInputParserProviderPath(void);
			};
		}
	}
}

