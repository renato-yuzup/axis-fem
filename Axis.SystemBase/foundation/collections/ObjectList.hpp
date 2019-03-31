#pragma once
#include "Collectible.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace foundation
	{
		namespace collections
		{
			/**********************************************************************************************//**
			 * @brief	Defines a list of collectible objects.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API ObjectList
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~ObjectList(void);

				/**********************************************************************************************//**
				 * @brief	Creates a new list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A new empty list object.
				 **************************************************************************************************/
				static ObjectList& Create(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Adds an object to the end of the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	object	The object to add.
				 **************************************************************************************************/
				virtual void Add(Collectible& object) = 0;

				/**********************************************************************************************//**
				 * @brief	Queries if an object exists in the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	object	The object to check.
				 *
				 * @return	true if the object is in this collection, false otherwise.
				 **************************************************************************************************/
				virtual bool Contains(Collectible& object) const = 0;

				/**********************************************************************************************//**
				 * @brief	Removes the given object from the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	object	The object to remove.
				 **************************************************************************************************/
				virtual void Remove(Collectible& object) = 0;

				/**********************************************************************************************//**
				 * @brief	Empties this list.
				 * 			@remark Objects stored in this list are not 
				 * 					freed in memory.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Clear(void) = 0;

				/**********************************************************************************************//**
				 * @brief	Returns the object stored in a specific position in the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	index	Zero-based index of the position to lookup.
				 *
				 * @return	A reference to the object stored in that position.
				 **************************************************************************************************/
				virtual Collectible& Get(size_t index) const = 0;

				/**********************************************************************************************//**
				 * <summary> Returns the index of an object in the collection.</summary>
				 *
				 * <param name="item"> [in,out] The item.</param>
				 *
				 * <returns> The zero-based index of the object.</returns>
				 **************************************************************************************************/
				virtual size_t GetIndex(Collectible& item) const = 0;

				/**********************************************************************************************//**
				 * @brief	Returns the object stored in a specific position in the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	index	Zero-based index of the position to lookup.
				 *
				 * @return	A reference to the object stored in that position.
				 **************************************************************************************************/
				virtual Collectible& operator [](size_t index) const = 0;

				/**********************************************************************************************//**
				 * @brief	Returns the number of items in this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-negative number representing the element count 
				 * 			in this collection.
				 **************************************************************************************************/
				virtual size_type Count(void) const = 0;
			};
		}
	}
}

