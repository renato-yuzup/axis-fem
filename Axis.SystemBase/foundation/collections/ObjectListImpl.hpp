#pragma once
#include "ObjectList.hpp"
#include <vector>

namespace axis
{
	namespace foundation
	{
		namespace collections
		{
			/**********************************************************************************************//**
			 * @brief	Implements a list of collectible objects.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 *
			 * @sa	ObjectList
			 **************************************************************************************************/
			class ObjectListImpl : public ObjectList
			{
			private:
				typedef std::vector<Collectible *> list;
				list _objects;
			public:

				/**********************************************************************************************//**
				 * @brief	Creates a new list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				ObjectListImpl(void);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~ObjectListImpl(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const;

				/**********************************************************************************************//**
				 * @brief	Adds an object to the end of the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	object	The object to add.
				 **************************************************************************************************/
				virtual void Add(Collectible& object);

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
				virtual bool Contains(Collectible& object) const;

				/**********************************************************************************************//**
				 * @brief	Removes the given object from the list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	object	The object to remove.
				 **************************************************************************************************/
				virtual void Remove(Collectible& object);

				/**********************************************************************************************//**
				 * @brief	Empties this list.
				 * 			@remark Objects stored in this list are not 
				 * 					freed in memory.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Clear(void);

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
				virtual Collectible& Get(size_t index) const;

				/**********************************************************************************************//**
				 * <summary> Returns the index of an object in the collection.</summary>
				 *
				 * <param name="item"> [in,out] The item.</param>
				 *
				 * <returns> The zero-based index of the object.</returns>
				 **************************************************************************************************/
				virtual size_t GetIndex(Collectible& item) const;

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
				virtual Collectible& operator [](size_t index) const;

				/**********************************************************************************************//**
				 * @brief	Returns the number of items in this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-negative number representing the element count 
				 * 			in this collection.
				 **************************************************************************************************/
				virtual size_type Count(void) const;
			};
		}
	}
}

