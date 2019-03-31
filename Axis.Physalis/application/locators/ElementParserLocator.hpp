#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/factories/parsers/ElementParserFactory.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/core/SectionDefinition.hpp"

namespace axis { namespace application { namespace locators {

class AXISPHYSALIS_API ElementParserLocator : 
  public axis::application::factories::parsers::BlockProvider
{
public:
  ElementParserLocator(void);
	~ElementParserLocator(void);
	void RegisterFactory(axis::application::factories::parsers::ElementParserFactory& factory);
	void UnregisterFactory(axis::application::factories::parsers::ElementParserFactory& factory);
  bool CanBuildElement(
    const axis::application::parsing::core::SectionDefinition& sectionDefinition) const;
	virtual axis::application::parsing::parsers::BlockParser& BuildParser(const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition, 
    axis::domain::collections::ElementSet& elementCollection ) const;

	/**********************************************************************************************//**
		* @fn	virtual axis::application::Input::parsers::Base::BlockParser& ElementParserLocator::BuildVoidParser(void) const = 0;
		*
		* @brief	Builds a block parser which consumes any provided contents without throwing an error (like a void).
		*
		* @author	Renato T. Yamassaki
		* @date	16 abr 2012
		*
		* @return	The void parser.
		**************************************************************************************************/
	axis::application::parsing::parsers::BlockParser& BuildVoidParser(void) const;

  virtual void axis::application::locators::ElementParserLocator::UnregisterProvider(
    axis::application::factories::parsers::BlockProvider& provider);
  virtual void axis::application::locators::ElementParserLocator::RegisterProvider(
    axis::application::factories::parsers::BlockProvider& provider);
  virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
  virtual bool IsLeaf( void );
  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};			

} } } // namespace axis::application:locators
