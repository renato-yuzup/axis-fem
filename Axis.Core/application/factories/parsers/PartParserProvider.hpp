#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "application/locators/MaterialFactoryLocator.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class PartParserProvider : public axis::application::factories::parsers::BlockProvider
{
public:
	PartParserProvider(void);
	virtual ~PartParserProvider(void);

	virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& params );

	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& params );

	/**********************************************************************************************//**
		* @fn	virtual const char Provider::*GetFeaturePath(void) const = 0;
		*
		* @brief	Returns the fully qualified feature path to which this
		* 			provider is identified.
		*
		* @author	Renato T. Yamassaki
		* @date	11 abr 2012
		*
		* @return	null if it fails, else the feature path.
		**************************************************************************************************/
	virtual const char *GetFeaturePath(void) const;

	/**********************************************************************************************//**
		* @fn	virtual const char Provider::*GetFeatureName(void) const = 0;
		*
		* @brief	Gets the feature name.
		*
		* @author	Renato T. Yamassaki
		* @date	10 mar 2011
		*
		* @return	The feature name.
		**************************************************************************************************/
	virtual const char *GetFeatureName(void) const;
protected:
	virtual void DoOnPostProcessRegistration( 
    axis::services::management::GlobalProviderCatalog& rootManager );
private:
  axis::application::locators::ElementParserLocator *_elementProvider;
  axis::application::locators::MaterialFactoryLocator *_modelProvider;
};

} } } } // namespace axis::application::factories::parsers
