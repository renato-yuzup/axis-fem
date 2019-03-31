#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/management/Provider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

/**********************************************************************************************//**
	* @class	BlockProvider
	*
	* @brief	Represents a provider for input file block parsing.
	*
	* @author	Renato T. Yamassaki
	* @date	14 mar 2011
	**************************************************************************************************/
class AXISPHYSALIS_API BlockProvider  : public axis::services::management::Provider
{
public:
	BlockProvider(void);
	virtual ~BlockProvider(void);

	/**********************************************************************************************//**
		* @fn	virtual bool abstract::CanParse(const axis::String& blockName,
		* 		axis::application::Input::evaluation::ParameterList& params) = 0;
		*
		* @brief	Queries if we can parse 'blockName' with the specified list of parameters.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param	blockName	  	Name of the block.
		* @param [in,out]	params	Optional and required parameters for the block.
		*
		* @return	true if this provider can parse this block, false otherwise.
		**************************************************************************************************/
	virtual bool CanParse(const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList) = 0;

	/**********************************************************************************************//**
		* @fn	virtual axis::application::Input::parsers::BlockParser& :::BuildParser(axis::String& contextName,
		* 		ParameterList& params) = 0;
		*
		* @brief	Gets a new block parser.
		*
		* @author	Renato T. Yamassaki
		* @date	25 mar 2011
		*
		* @param [in,out]	contextName	Name of the context.
		* @param [in,out]	params	   	Options for controlling the operation.
		*
		* @return	The created parser.
		**************************************************************************************************/
	virtual axis::application::parsing::parsers::BlockParser& BuildParser(const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList) = 0;

	/**********************************************************************************************//**
		* @fn	bool :::ContainsProvider(const axis::String& blockName,
		* 		const axis::services::language::syntax::evaluation::ParameterList& paramList) const;
		*
		* @brief	Query if this provider directly knows another provider
		* 			capable of parsing 'blockName'.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param	blockName	Name of the block.
		* @param	paramList	Optional and required parameters for the
		* 						block.
		*
		* @return	true if it knows, false otherwise.
		**************************************************************************************************/
	bool ContainsProvider(const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList) const;

	/**********************************************************************************************//**
		* @fn	virtual void abstract::RegisterProvider(BlockProvider& provider);
		*
		* @brief	Registers a new subprovider.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param [in,out]	provider	The provider.
		**************************************************************************************************/
	void RegisterProvider(BlockProvider& provider);

	/**********************************************************************************************//**
		* @fn	virtual void abstract::UnregisterProvider(BlockProvider& provider);
		*
		* @brief	Unregisters an existing subprovider.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param [in,out]	provider	The provider.
		**************************************************************************************************/
	void UnregisterProvider(BlockProvider& provider);

	/**********************************************************************************************//**
		* @fn	virtual bool abstract::IsRegisteredProvider(const BlockProvider& provider);
		*
		* @brief	Query if 'provider' is registered in this object.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param	provider	The provider to test.
		*
		* @return	true if it is a registered provider, false otherwise.
		**************************************************************************************************/
	bool IsRegisteredProvider(BlockProvider& provider);

	/**********************************************************************************************//**
		* @fn	BlockProvider& :::GetProvider(const axis::String& blockName,
		* 		const axis::services::language::syntax::evaluation::ParameterList& paramList) const;
		*
		* @brief	Returns a provider capable of handling the specified
		* 			block.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param	blockName	Name of the block.
		* @param	paramList	Block parameters.
		*
		* @return	The capable provider.
		**************************************************************************************************/
	BlockProvider& GetProvider(const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList) const;

	/**********************************************************************************************//**
		* @fn	virtual bool abstract::IsLeaf(void);
		*
		* @brief	Returns if this object accepts subprovider registration.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @return	true if this object is a leaf and can't accept subproviders, false otherwise.
		**************************************************************************************************/
	virtual bool IsLeaf(void);

	/**********************************************************************************************//**
		* @fn	virtual void abstract::OnRegister(BlockProvider& parent);
		*
		* @brief	Execute post-registration actions necessary for the correct operation of this provider.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param [in,out]	parent	The parent provider to which this object is registered.
		**************************************************************************************************/
	virtual void OnRegister(BlockProvider& parent);

	/**********************************************************************************************//**
		* @fn	virtual void abstract::OnUnregister(BlockProvider& parent);
		*
		* @brief	Executes post-unregister actions to correctly tidy up changes made by this provider.
		*
		* @author	Renato T. Yamassaki
		* @date	14 mar 2011
		*
		* @param [in,out]	parent	The parent provider to which this object was registered.
		**************************************************************************************************/
	virtual void OnUnregister(BlockProvider& parent);

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
	* @fn	virtual void abstract::UnloadModule(GlobalProviderCatalog& manager);
	*
	* @brief	Unloads this module.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @param [in]	manager	The module manager to which this provider is registered.
	**************************************************************************************************/
	virtual void UnloadModule(axis::services::management::GlobalProviderCatalog& manager);
protected:
  virtual void DoOnPostProcessRegistration(axis::services::management::GlobalProviderCatalog& rootManager);
  virtual void DoOnUnload(axis::services::management::GlobalProviderCatalog& rootManager);

  axis::services::management::GlobalProviderCatalog *_rootManager;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } } // namespace axis::application::factories::parsers
