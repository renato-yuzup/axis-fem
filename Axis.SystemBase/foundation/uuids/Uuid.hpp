#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace foundation
	{
		namespace uuids
		{
			/**********************************************************************************************//**
			 * @brief	Stores an Universally Unique Identifier (UUID).
			 *
			 * @author	Renato T. Yamassaki
			 * @date	27 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Uuid
			{
			public:		// static members

				/**********************************************************************************************//**
				 * @brief	Defines the byte data type.
				 **************************************************************************************************/
				typedef unsigned char byte;

				/**********************************************************************************************//**
				 * @summary	The length, in bytes, of a UUID.
				 **************************************************************************************************/
				static const int Length;

				/**********************************************************************************************//**
				 * @brief	Generates a new random UUID.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	A UUID object.
				 **************************************************************************************************/
				static Uuid GenerateRandom(void);
			private:
				byte _bytes[16];
			public:

				/**********************************************************************************************//**
				 * @brief	Creates a new UUID with zeroed bytes.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 **************************************************************************************************/
				Uuid(void);

				/**********************************************************************************************//**
				 * @brief	Creates a new UUID from an array of bytes.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The array of bytes containing the
				* 							bytes to build the identifier.
				 **************************************************************************************************/
				Uuid(const byte * const uuid);

				/**********************************************************************************************//**
				 * @brief	Creates a new UUID from an array.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param uuid	Pointer to an array of integers
				 * 							from which the identifier will be built.
				 * 							Each position will be truncated to a byte.
				 **************************************************************************************************/
				Uuid(const int * const uuid);

				/**********************************************************************************************//**
				 * @brief	Creates a new UUID from an array.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param uuid	Pointer to an array of integers
				 * 							from which the identifier will be built.
				 * 							Each position will be truncated to a byte.
				 **************************************************************************************************/
				Uuid(const unsigned int * const uuid);

				/**********************************************************************************************//**
				 * @brief	Copy constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID object.
				 **************************************************************************************************/
				Uuid(const Uuid& uuid);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 **************************************************************************************************/
				~Uuid(void);

				/**********************************************************************************************//**
				 * @brief	Returns the string representation of this UUID.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	A string representation of the UUID in the format
				 * 			{00000000-0000-0000-0000-000000000000} (with braces).
				 **************************************************************************************************/
				axis::String ToString(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the string representation of this UUID.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	A string representation of the UUID in the format
				 * 			00000000-0000-0000-0000-000000000000 (no braces).
				 **************************************************************************************************/
				axis::String ToStringUnbraced(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the string representation of this UUID as an
				 * 			array of bytes.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	A string representation of the UUID in the format
				 * 			00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00.
				 **************************************************************************************************/
				axis::String ToStringAsByteArray(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the string representation of this UUID as a
				 * 			sequence of bytes.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	A string representation of the UUID in the format
				 * 			00000000000000000000000000000000.
				 **************************************************************************************************/
				axis::String ToStringAsByteSequence(void) const;

				/**********************************************************************************************//**
				 * @brief	Compares if this UUID is equal to another.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	true if both UUIDs are considered equivalent.
				 **************************************************************************************************/
				bool operator == (const Uuid& uuid) const;

				/**********************************************************************************************//**
				 * @brief	Compares if this UUID is different to another.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	true if both UUIDs are considered different.
				 **************************************************************************************************/
				bool operator != (const Uuid& uuid) const;

				/**********************************************************************************************//**
				 * @brief	Compares if this UUID should be ranked before another
				 * 			UUID in a sorted list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	true if this UUID is less than the other.
				 **************************************************************************************************/
				bool operator < (const Uuid& uuid) const;

				/**********************************************************************************************//**
				 * @brief	Compares if this UUID should be ranked after another
				 * 			UUID in a sorted list.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	true if this UUID is greater than the other.
				 **************************************************************************************************/
				bool operator > (const Uuid& uuid) const;

				/**********************************************************************************************//**
				 * @brief	Compares if this UUID should be ranked before another
				 * 			UUID in a sorted list or is at least equal.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	true if this UUID is less than the other.
				 **************************************************************************************************/
				bool operator <= (const Uuid& uuid) const;

				/**********************************************************************************************//**
				 * @brief	Compares if this UUID should be ranked after another
				 * 			UUID in a sorted list or is at least equal.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	true if this UUID is less than the other.
				 **************************************************************************************************/
				bool operator >= (const Uuid& uuid) const;

				/**********************************************************************************************//**
				 * @brief	Returns the index value at the specified position.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	index	Zero-based index of the byte.
				 *
				 * @return	The byte value stored in that position.
				 **************************************************************************************************/
				byte operator [](int index) const;

				/**********************************************************************************************//**
				 * @brief	Returns the index value at the specified position.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	index	Zero-based index of the byte.
				 *
				 * @return	The byte value stored in that position.
				 **************************************************************************************************/
				byte GetByte(int index) const;

				/**********************************************************************************************//**
				 * @brief	Assignment operator.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	uuid	The other UUID.
				 *
				 * @return	A reference to this object.
				 **************************************************************************************************/
				Uuid& operator =(const Uuid& uuid);
			};
		
      AXISSYSTEMBASE_API size_t hash_value(const Uuid& uuid);
		}
	}
}

