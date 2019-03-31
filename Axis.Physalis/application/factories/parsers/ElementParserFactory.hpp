#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/collections/ElementSet.hpp"
#include "application/parsing/parsers/BlockParser.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

/**********************************************************************************************//**
	* @class	ElementParserFactory
	*
	* @brief	Defines a specialized factory to build finite element objects.
	*
	* @author	Renato T. Yamassaki
	* @date	10 abr 2012
	**************************************************************************************************/
class AXISPHYSALIS_API ElementParserFactory
{
public:
	virtual ~ElementParserFactory(void);

	/**********************************************************************************************//**
		* @fn	virtual bool ElementParserFactory::CanBuild(const axis::domain::analyses::SectionDefinition& definition) const = 0;
		*
		* @brief	Queries if we can build a finite element with the specified section definition.
		*
		* @author	Renato T. Yamassaki
		* @date	28 mar 2011
		*
		* @param	definition	Parameters needed to build the element.
		*
		* @return	true if we can, false otherwise.
		*
		* ### param [in,out]	params	Options for controlling the operation.
		**************************************************************************************************/
  virtual bool CanBuild(
    const axis::application::parsing::core::SectionDefinition& definition) const = 0;

	/**********************************************************************************************//**
		* @fn	virtual axis::application::Input::parsers::Base::BlockParser& ElementParserFactory::BuildParser(const axis::domain::analyses::SectionDefinition& definition,
		* 		axis::domain::collections::ElementCollection& elementCollection) const = 0;
		*
		* @brief	Builds a finite element parser.
		*
		* @author	Renato T. Yamassaki
		* @date	28 mar 2011
		*
		* @param	definition				 	The finite element
		* 										specifications.
		* @param [in,out]	elementCollection	Element collection to which
		* 										the object is going to be
		* 										added.
		*
		* @return	A block parser able to build the specified finite element.
		**************************************************************************************************/
	virtual axis::application::parsing::parsers::BlockParser& BuildParser(
    const axis::application::parsing::core::SectionDefinition& definition, 
    axis::domain::collections::ElementSet& elementCollection) const = 0;

	/**********************************************************************************************//**
		* @fn	virtual void ElementParserFactory::Destroy(void) const = 0;
		*
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	16 abr 2012
		**************************************************************************************************/
	virtual void Destroy(void) const = 0;
};		

} } } } // namespace axis::application::factories::parsers
