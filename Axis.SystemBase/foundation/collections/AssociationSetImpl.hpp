#pragma once
#include "foundation/collections/AssociationSet.hpp"
#include "boost/multi_index_container.hpp"
#include "boost/multi_index/sequenced_index.hpp"
#include "boost/multi_index/ordered_index.hpp"
#include "boost/multi_index/hashed_index.hpp"
#include "boost/multi_index/member.hpp"
#include "boost/multi_index/random_access_index.hpp"
#include <map>

namespace axis
{
	namespace foundation
	{
		namespace collections
		{
			/**********************************************************************************************//**
			 * @brief	Implements a collection of unique keys that each one maps
			 * 			to an object.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 *
			 * @sa	AssociationSet
			 **************************************************************************************************/
			class AssociationSetImpl : public AssociationSet
			{
			public:
				struct mutable_pair
				{
				public:
					typedef key_type first_type;
					typedef value_type * second_type;

					mutable_pair(void);
					mutable_pair(first_type& f,const second_type& s);

					first_type first;
					mutable second_type second;
				};
			private:

				// typedef std::map<key_type, value_type *> collection;
				typedef boost::multi_index::random_access<>  numbered_index;
				typedef boost::multi_index::hashed_unique<boost::multi_index::member<mutable_pair,key_type,&mutable_pair::first>> unique_index;
				typedef boost::multi_index::indexed_by<unique_index, numbered_index> index;
				typedef boost::multi_index::multi_index_container<
							mutable_pair,
							index
						> collection;

				collection _objects;
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
					collection::const_iterator _current;
					collection::const_iterator _end;
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
					IteratorLogicImpl(const collection::const_iterator& current, const collection::const_iterator& end);

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
					virtual value_type& GetItem(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns the key of current value.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	The key.
					 **************************************************************************************************/
					virtual key_type GetKey(void) const;

					/**********************************************************************************************//**
					 * @brief	Indirection operator.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	25 ago 2012
					 *
					 * @return	A pointer to the value stored in current position.
					 **************************************************************************************************/
					virtual value_type& operator *(void) const;

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
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~AssociationSetImpl(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const;

				/**********************************************************************************************//**
				 * @brief	Adds a new mapped key-value pair to the collection.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id		   	The unique identifier for the object.
				 * @param [in,out]	obj	The object to store.
				 **************************************************************************************************/
				virtual void Add(const_key_type id, value_type& obj);

				/**********************************************************************************************//**
				 * @brief	Removes an object by its key.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	id	The object key.
				 **************************************************************************************************/
				virtual void Remove(const_key_type id);

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
				virtual value_type& Get(const_key_type id) const;

				/**********************************************************************************************//**
				 * <summary> Returns the object in the specified position.</summary>
				 *
				 * <param name="id"> The zero-based index of the object to retrieve.</param>
				 *
				 * <returns> A reference to the object.</returns>
				 **************************************************************************************************/
				virtual value_type& Get( size_type id ) const;

				/**********************************************************************************************//**
				 * <summary> Returns key of the object in the specified position.</summary>
				 *
				 * <param name="id"> The zero-based index of the object to query.</param>
				 *
				 * <returns> The object key.</returns>
				 **************************************************************************************************/
				virtual key_type GetKey( size_type id ) const;

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
				virtual value_type& operator [](const_key_type id) const;

				/**********************************************************************************************//**
				 * <summary> Returns the object in the specified position.</summary>
				 *
				 * <param name="id"> The zero-based index of the object to retrieve.</param>
				 *
				 * <returns> A reference to the object.</returns>
				 **************************************************************************************************/
				virtual value_type& operator []( size_type id ) const;

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
				virtual bool Contains(const_key_type id) const;

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

				/**********************************************************************************************//**
				 * @brief	Empties this collection and releases resources used
				 * 			by stored items.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void DestroyChildren(void);

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

