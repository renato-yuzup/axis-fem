#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/factories/elements/NodeFactory.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class NodeParserProvider : public axis::application::factories::parsers::BlockProvider
{
protected:
	/**********************************************************************************************//**
	* @fn	virtual void abstract::DoOnPostProcessRegistration(GlobalProviderCatalog& manager);
	*
	* @brief	Executes post processing tasks after registration. This operation should roll back every
	* 			change made to the system if an error occurs.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @param [in]	manager	The module manager to which this provider was registered.
	**************************************************************************************************/
	virtual void DoOnPostProcessRegistration(axis::services::management::GlobalProviderCatalog& manager);

	/**********************************************************************************************//**
	* @fn	virtual void abstract::DoOnUnload(GlobalProviderCatalog& manager);
	*
	* @brief	Unloads this module.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @param [in]	manager	The module manager to which this provider is registered.
	**************************************************************************************************/
	virtual void DoOnUnload(axis::services::management::GlobalProviderCatalog& manager);
public:
	NodeParserProvider(void);
	virtual ~NodeParserProvider(void);

	/**********************************************************************************************//**
	* @fn	virtual bool abstract::CanParse(const axis::String& blockName,
	* 		axis::application::parsing::evaluation::ParameterList& params);
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
    const axis::services::language::syntax::evaluation::ParameterList& params);

	/**********************************************************************************************//**
	* @fn	virtual const char :::*GetFeaturePath(void) const;
	*
	* @brief	Returns the fully qualified feature path to which this provider is identified.
	*
	* @author	Renato T. Yamassaki
	* @date	10 mar 2011
	*
	* @return	The feature path.
	**************************************************************************************************/
	virtual const char *GetFeaturePath(void) const;

	/**********************************************************************************************//**
	* @fn	virtual const char abstract::*GetFeatureName(void) const;
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
	* @fn	virtual axis::application::parsing::parsers::BlockParser& BlockProvider::BuildParser(void) = 0;
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
    const axis::services::language::syntax::evaluation::ParameterList& params);
private:
  axis::application::factories::elements::NodeFactory *_nodeFactory;
};	

} } } } // namespace axis::application::factories::parsers
