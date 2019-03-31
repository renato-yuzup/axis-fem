#pragma once
#include "foundation/collections/Collectible.hpp"
#include "AxisString.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace foundation
	{
		namespace collections
		{
			/**********************************************************************************************//**
			 * @brief	Defines a collection of unique keys that each one maps 
			 * 			to an object.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API ObjectMap
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Defines the strategy for the iterator object for this type.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API IteratorLogic
				{
				public:

					/**********************************************************************************************//**
					 * @brief	Destroys this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 **************************************************************************************************/
					virtual void Destroy(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Queries if there is another value to iterate.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	true if there is a next value, false otherwise.
					 **************************************************************************************************/
					virtual bool HasNext(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Go to the next value.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 **************************************************************************************************/
					virtual void GoNext(void) = 0;

					/**********************************************************************************************//**
					 * @brief	Returns the key of current value.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	The key.
					 **************************************************************************************************/
					virtual axis::String GetKey(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Returns the value stored in current position.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	The item.
					 **************************************************************************************************/
					virtual Collectible& GetItem(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Indirection operator.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the value stored in current position.
					 **************************************************************************************************/
					virtual Collectible& operator *(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					virtual IteratorLogic& Clone(void) const = 0;
				};

				/**********************************************************************************************//**
				 * @brief	A forward-only iterator for this collection type.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API Iterator
				{
				private:
					IteratorLogic *_logic;

					/**********************************************************************************************//**
					 * @brief	Creates a new instance of this iterator.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param [in,out]	logic	The strategy that this object will use.
					 **************************************************************************************************/
					Iterator(IteratorLogic& logic);
				public:

					/**********************************************************************************************//**
					 * @brief	Copy constructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param	other	The source object.
					 **************************************************************************************************/
					Iterator(const Iterator& other);

					/**********************************************************************************************//**
					 * @brief	Copy assignment operator.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param	other	The source object.
					 *
					 * @return	A reference to this object.
					 **************************************************************************************************/
					Iterator& operator =(const Iterator& other);

					/**********************************************************************************************//**
					 * @brief	Queries if there is another value to iterate.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	true if there is a next value, false otherwise.
					 **************************************************************************************************/
					virtual bool HasNext(void) const;

					/**********************************************************************************************//**
					 * @brief	Go to the next value.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 **************************************************************************************************/
					virtual void GoNext(void);

					/**********************************************************************************************//**
					 * @brief	Returns the value stored in current position.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	The item.
					 **************************************************************************************************/
					virtual Collectible& GetItem(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns the key of current value.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	The key.
					 **************************************************************************************************/
					virtual axis::String GetKey(void) const;

					/**********************************************************************************************//**
					 * @brief	Indirection operator.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the value stored in current position.
					 **************************************************************************************************/
					virtual Collectible& operator *(void) const;
					
					friend class ObjectMap;
				};

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~ObjectMap(void);

				/**********************************************************************************************//**
				 * @brief	Creates a new collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	An empty collection object.
				 **************************************************************************************************/
				static ObjectMap& Create(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Adds a new mapped key-value pair to the collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id		   	The unique identifier for the object.
				 * @param [in,out]	obj	The object to store.
				 **************************************************************************************************/
				virtual void Add(const axis::String& id, Collectible& obj) = 0;

				/**********************************************************************************************//**
				 * @brief	Removes an object by its key.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id	The object key.
				 **************************************************************************************************/
				virtual void Remove(const axis::String& id) = 0;

				/**********************************************************************************************//**
				 * @brief	Returns the object with the specified key.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id	The object key.
				 *
				 * @return	A reference to the object.
				 **************************************************************************************************/
				virtual Collectible& Get(const axis::String& id) const = 0;

				/**********************************************************************************************//**
				 * @brief	Returns the object with the specified key.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id	The object key.
				 *
				 * @return	A reference to the object.
				 **************************************************************************************************/
				virtual Collectible& operator [](const axis::String& id) const = 0;

				/**********************************************************************************************//**
				 * @brief	Quries if there is an object associated with the specified key.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id	The object key.
				 *
				 * @return	true if key is found, false otherwise.
				 **************************************************************************************************/
				virtual bool Contains(const axis::String& id) const = 0;

				/**********************************************************************************************//**
				 * @brief	Empties this collection.
				 * 			@remark Items stored in this collection are not
				 * 					freed in memory.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Clear(void) = 0;

				/**********************************************************************************************//**
				 * @brief	Returns the number of items stored in this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-negative number representing the element count.
				 **************************************************************************************************/
				virtual size_type Count(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Empties this collection and releases resources used
				 * 			by stored items.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void DestroyChildren(void) = 0;

				/**********************************************************************************************//**
				 * @brief	Returns an iterator for this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	The iterator object.
				 **************************************************************************************************/
				Iterator GetIterator(void) const;
			protected:

				/**********************************************************************************************//**
				 * @brief	Returns an appropriate strategy for use by iterator
				 * 			objects of this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A reference to a new iterator strategy object.
				 **************************************************************************************************/
				virtual IteratorLogic& DoGetIterator(void) const = 0;
			};
		}
	}
}

