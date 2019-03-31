#pragma once
#include "services/management/Provider.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "domain/elements/Node.hpp"

namespace axis { namespace application { namespace factories { namespace elements {

class NodeFactory : public axis::services::management::Provider
{
public:
	NodeFactory(void);
	~NodeFactory(void);

	axis::foundation::memory::RelativePointer CreateNode(axis::domain::elements::Node::id_type userId);
	axis::foundation::memory::RelativePointer CreateNode(axis::domain::elements::Node::id_type userId, 
    coordtype x, coordtype y);
	axis::foundation::memory::RelativePointer CreateNode(axis::domain::elements::Node::id_type userId, 
    coordtype x, coordtype y, coordtype z);

	/**********************************************************************************************//**
		* @fn	virtual const char :::*GetFeaturePath(void) const = 0;
		*
		* @brief	Returns the fully qualified feature path to which this provider is identified.
		*
		* @author	Renato T. Yamassaki
		* @date	22 mar 2011
		*
		* @return	null if it fails, else the feature path.
		**************************************************************************************************/
	virtual const char *GetFeaturePath(void) const;

	/**********************************************************************************************//**
	* @fn	virtual const char abstract::*GetFeatureName(void) const = 0;
	*
	* @brief	Gets the feature name.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @return	The feature name.
	**************************************************************************************************/
	virtual const char *GetFeatureName(void) const;

	/**********************************************************************************************//**
	* @fn	virtual void abstract::PostProcessRegistration(GlobalProviderCatalog& manager);
	*
	* @brief	Executes post processing tasks after registration. This operation should roll back every
	* 			change made to the system if an error occurs.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @param [in]	manager	The module manager to which this provider was registered.
	**************************************************************************************************/
	virtual void PostProcessRegistration(axis::services::management::GlobalProviderCatalog& manager);

	/**********************************************************************************************//**
	* @fn	virtual void abstract::UnloadModule(GlobalProviderCatalog& manager) = 0;
	*
	* @brief	Unloads this module.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @param [in]	manager	The module manager to which this provider is registered.
	**************************************************************************************************/
	virtual void UnloadModule(axis::services::management::GlobalProviderCatalog& manager);
private:
  axis::domain::elements::Node::id_type _nextId;
};

} } } } // namespace axis::application::factories::elements

