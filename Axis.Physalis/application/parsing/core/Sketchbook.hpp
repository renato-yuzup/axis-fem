#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "SectionDefinition.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

/**********************************************************************************************//**
	* @class	Sketchbook
	*
	* @brief	Defines a place to save metadata needed during the
	* 			model building process. 
	*
	* @author	Renato T. Yamassaki
	* @date	10 abr 2012
	**************************************************************************************************/
class AXISPHYSALIS_API Sketchbook
{
public:
  Sketchbook(void);

	/**********************************************************************************************//**
		* @fn	virtual Sketchbook::~Sketchbook(void);
		*
		* @brief	Destructor.
		*
		* @author	Renato T. Yamassaki
		* @date	10 abr 2012
		**************************************************************************************************/
	~Sketchbook(void);

	/**********************************************************************************************//**
		* @fn	virtual void Sketchbook::Destroy(void) const = 0;
		*
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		**************************************************************************************************/
	void Destroy(void) const;

	/**********************************************************************************************//**
		* @fn	virtual bool Sketchbook::HasSectionDefined(const axis::String& setId) const = 0;
		*
		* @brief	Query if the element set identified by 'setId' has the section defined.
		*
		* @author	Renato T. Yamassaki
		* @date	10 abr 2012
		*
		* @param	setId	Identifier for the set.
		*
		* @return	true if section defined, false if not.
		**************************************************************************************************/
	bool HasSectionDefined(const axis::String& setId) const;

	/**********************************************************************************************//**
		* @fn	virtual void Sketchbook::AddSection(const axis::String& setId,
		* 		const SectionDefinition& sectionDefinition) = 0;
		*
		* @brief	Adds a section definition for the element set 'setId'.
		*
		* @author	Renato T. Yamassaki
		* @date	10 abr 2012
		*
		* @param	setId			 	Identifier for the set.
		* @param	sectionDefinition	The section definition.
		**************************************************************************************************/
	void AddSection(const axis::String& setId, const SectionDefinition& sectionDefinition);

	/**********************************************************************************************//**
		* @fn	virtual const SectionDefinition& Sketchbook::RemoveSection(const axis::String& setId) = 0;
		*
		* @brief	Removes the section definition for the element set 'setId'.
		*
		* @author	Renato T. Yamassaki
		* @date	10 abr 2012
		*
		* @param	setId	Identifier for the set.
		*
		* @return	The section definition removed.
		**************************************************************************************************/
	const SectionDefinition& RemoveSection(const axis::String& setId);

	/**********************************************************************************************//**
		* @fn	virtual const SectionDefinition& Sketchbook::GetSection(const axis::String& setId) const = 0;
		*
		* @brief	Gets the section defined for the element set 'setId'.
		*
		* @author	Renato T. Yamassaki
		* @date	10 abr 2012
		*
		* @param	setId	Identifier for the set.
		*
		* @return	The section definition.
		**************************************************************************************************/
	const SectionDefinition& GetSection(const axis::String& setId) const;

	/**********************************************************************************************//**
		* @fn	virtual unsigned int Sketchbook::SectionsCount(void) const = 0;
		*
		* @brief	Gets the section definitions count.
		*
		* @author	Renato T. Yamassaki
		* @date	10 abr 2012
		*
		* @return	An unsigned integer representing how many definitions are stored.
		**************************************************************************************************/
	unsigned int SectionsCount(void) const;
private:
  class Pimpl;
  Sketchbook(const Sketchbook&);
  Sketchbook& operator =(const Sketchbook&);
  Pimpl *pimpl_;
};

} } } } // namespace axis::application::parsing::core
