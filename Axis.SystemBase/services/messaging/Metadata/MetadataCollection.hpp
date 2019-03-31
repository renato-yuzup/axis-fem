#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/metadata/Metadatum.hpp"
#include "foundation/collections/ObjectMap.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			namespace metadata
			{
				/**********************************************************************************************//**
				 * @brief	Stores a collection of metadata objects.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 * @sa axis::services::messaging::metadata::Metadatum
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API MetadataCollection
				{
				private:
					axis::foundation::collections::ObjectMap &_metadata;

					/**********************************************************************************************//**
					 * @brief	Private constructor. Static method Create() should be
					 * 			used instead.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					MetadataCollection(void);
				public:

					/**********************************************************************************************//**
					 * @brief	Destructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					~MetadataCollection(void);

					/**********************************************************************************************//**
					 * @brief	Creates a new instance of this class.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A new MetadataCollection object.
					 **************************************************************************************************/
					static MetadataCollection& Create(void);

					/**********************************************************************************************//**
					 * @brief	Adds a metadatum object to the collection.
					 * 			@remark  Only one instance of an object of a 
					 * 					 derived type of Metadatum can be added 
					 * 					 to a collection at a time. Also, note 
					 * 					 that a clone of the specified object 
					 * 					 is added to the collection, so that
					 * 					 the passed reference can be destroyed
					 * 					 afterwards if needed.
					 * 					 
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	metadatum	The metadata to add.
					 **************************************************************************************************/
					void Add(const Metadatum& metadatum);

					/**********************************************************************************************//**
					 * @brief	Queries if there is a metadata object of the 
					 * 			specified type.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	typeName	The type name of the metadata object, as
					 * 						returned by the GetTypeName() member of the
					 * 						same object.
					 *
					 * @return	true if such an object exists, false otherwise.
					 **************************************************************************************************/
					bool Contains(const axis::String& typeName) const;

					/**********************************************************************************************//**
					 * @brief	Removes the given typeName.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	typeName	The type name of the metadata object 
					 * 						to remove.
					 **************************************************************************************************/
					void Remove(const axis::String& typeName);

					/**********************************************************************************************//**
					 * @brief	Returns a metadata object which its type name 
					 * 			is as specified.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	typeName	The type name of the object to 
					 * 						retrieve.
					 *
					 * @return	A reference to the metadata object.
					 **************************************************************************************************/
					axis::services::messaging::metadata::Metadatum& Get (const axis::String& typeName) const;

					/**********************************************************************************************//**
					 * @brief	Returns a metadata object which its type name 
					 * 			is as specified.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	typeName	The type name of the object to 
					 * 						retrieve.
					 *
					 * @return	A reference to the metadata object.
					 **************************************************************************************************/
					axis::services::messaging::metadata::Metadatum& operator [] (const axis::String& typeName) const;

					/**********************************************************************************************//**
					 * @brief	Clear the collection and release all resources.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					void Clear(void);

					/**********************************************************************************************//**
					 * @brief	Returns the number of elements in this collection. 
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A non-negative number representing the number 
					 * 			of elements.
					 **************************************************************************************************/
					size_type Count(void) const;

					/**********************************************************************************************//**
					 * @brief	Returns if this collection holds no elements.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	true if the collection is empty, false otherwise.
					 **************************************************************************************************/
					bool Empty(void) const;

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object. All objects in
					 * 			the collection are also cloned.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					MetadataCollection& Clone(void) const;

					/**********************************************************************************************//**
					 * @brief	Destroys this object and all its elements.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					void Destroy(void) const;
				};			
			}
		}
	}
}

