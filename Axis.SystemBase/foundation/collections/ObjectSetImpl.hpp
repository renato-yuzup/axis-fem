#pragma once
#include "ObjectSet.hpp"
#include <set>

namespace axis
{
	namespace foundation
	{
		namespace collections
		{
			/**********************************************************************************************//**
			 * @brief	Implements a collection of unique objects.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 *
			 * @sa	ObjectSet
			 **************************************************************************************************/
			class ObjectSetImpl : public ObjectSet
			{
			private:
				typedef std::set<Collectible *> list;
				list _objects;
			public:
				/**********************************************************************************************//**
				 * @brief	Defines the strategy for the iterator object for this type.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				class IteratorLogicImpl : public IteratorLogic
				{
				private:
					list::iterator _current;
					list::iterator _end;
				public:

					/**********************************************************************************************//**
					 * @brief	Constructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @param	current	The start position for the iterator.
					 * @param	end	   	The position which defines that no more 
					 * 					positions exists in the collection.
					 **************************************************************************************************/
					IteratorLogicImpl(const list::iterator& current, const list::iterator& end);

					/**********************************************************************************************//**
					 * @brief	Destroys this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 **************************************************************************************************/
					virtual void Destroy(void) const;

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
					 * @brief	Indirection operator.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the value stored in current position.
					 **************************************************************************************************/
					virtual Collectible& operator *(void) const;

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					virtual IteratorLogic& Clone(void) const;
				};

				/**********************************************************************************************//**
				 * @brief	Default constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				ObjectSetImpl(void);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~ObjectSetImpl(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const;

				/**********************************************************************************************//**
				 * @brief	Adds a new unique object to the collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	object	The object to store.
				 **************************************************************************************************/
				virtual void Add(Collectible& object);

				/**********************************************************************************************//**
				 * @brief	Quries if an object is stored in this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	object	The object to check.
				 *
				 * @return	true if the object is found, false otherwise.
				 **************************************************************************************************/
				virtual bool Contains(const Collectible& object) const;

				/**********************************************************************************************//**
				 * @brief	Removes an object from the collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	object	The object to remove.
				 **************************************************************************************************/
				virtual void Remove(const Collectible& object);

				/**********************************************************************************************//**
				 * @brief	Empties this collection.
				 * 			@remark Items stored in this collection are not
				 * 					freed in memory.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Clear(void);

				/**********************************************************************************************//**
				 * @brief	Returns the number of items stored in this collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A non-negative number representing the element count.
				 **************************************************************************************************/
				virtual size_type Count(void) const;

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
				virtual IteratorLogic& DoGetIterator( void ) const;
			};
		}
	}
}

