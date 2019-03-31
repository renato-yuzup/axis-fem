/**********************************************************************************************//**
 * @file	foundation/assertion/assertion.hpp
 *
 * @brief	Contains template methods that checks type names and casts types of
 * 			ascertainable objects.
 **************************************************************************************************/
#pragma once
#include "foundation/Assertion/Ascertainable.hpp"
#include "foundation/InvalidCastException.hpp"

namespace axis
{
	/**********************************************************************************************//**
	 * @brief	Checks if an object can be casted to the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @param	obj	The object to be verified.
	 *
	 * @return	true if cast is possible, false otherwise.
	 **************************************************************************************************/
	template<class T> bool is_type_of(const axis::foundation::assertion::Ascertainable& obj)
	{
		return (typename T::GetClassName() == obj.GetTypeName());
	}

	/**********************************************************************************************//**
	 * @brief	Checks if an object can be casted to the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @param	obj	The object to be verified.
	 *
	 * @return	true if cast is possible, false otherwise.
	 **************************************************************************************************/
	template<class T> bool is_type_of(const axis::foundation::assertion::Ascertainable *const obj)
	{
		return (typename T::GetClassName() == obj->GetTypeName());
	}

	/**********************************************************************************************//**
	 * @brief	Safely casts an object to the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @exception	axis::foundation::InvalidCastException	Thrown when
	 * 														the object
	 * 														cannot be
	 * 														casted to the
	 * 														specified
	 * 														type.
	 *
	 * @param	obj	The object to cast.
	 *
	 * @return	A reference to the object.
	 **************************************************************************************************/
	template<class T> const typename T& safe_cast(const axis::foundation::assertion::Ascertainable& obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			throw axis::foundation::InvalidCastException(_T("Cannot cast to an object of type ") + typename T::GetClassName());
		}
		return static_cast<const typename T&>(obj);
	}

	/**********************************************************************************************//**
	 * @brief	Safely casts an object to the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @exception	axis::foundation::InvalidCastException	Thrown when
	 * 														the object
	 * 														cannot be
	 * 														casted to the
	 * 														specified
	 * 														type.
	 *
	 * @param [in,out]	obj	The object to cast.
	 *
	 * @return	A reference to the object.
	 **************************************************************************************************/
	template<class T> typename T& safe_cast(axis::foundation::assertion::Ascertainable& obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			throw axis::foundation::InvalidCastException(_T("Cannot cast to an object of type ") + typename T::GetClassName());
		}
		return static_cast<typename T&>(obj);
	}

	/**********************************************************************************************//**
	 * @brief	Casts, if possible, the specified object to a pointer of
	 * 			the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @param	obj	The object to cast.
	 *
	 * @return	A pointer to the object or null if it fails.
	 **************************************************************************************************/
	template<class T> const typename T *cast_or_null(const axis::foundation::assertion::Ascertainable& obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			return NULL;
		}
		return static_cast<const typename T *>(&obj);
	}

	/**********************************************************************************************//**
	 * @brief	Casts, if possible, the specified object to a pointer of
	 * 			the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @param [in,out]	obj	The object to cast.
	 *
	 * @return	A pointer to the object or null if it fails.
	 **************************************************************************************************/
	template<class T> typename T *cast_or_null(axis::foundation::assertion::Ascertainable& obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			return NULL;
		}
		return static_cast<typename T *>(&obj);
	}

	/**********************************************************************************************//**
	 * @brief	Safely casts an object to the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @exception	axis::foundation::InvalidCastException	Thrown when
	 * 														the object
	 * 														cannot be
	 * 														casted to the
	 * 														specified
	 * 														type.
	 *
	 * @param	obj	Pointer to the object to cast.
	 *
	 * @return	A pointer to the object.
	 **************************************************************************************************/
	template<class T> typename T * const safe_cast(const axis::foundation::assertion::Ascertainable * const obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			throw axis::foundation::InvalidCastException(_T("Cannot cast to an object of type ") + typename T::GetClassName());
		}
		return static_cast<typename T * const>(obj);
	}

	/**********************************************************************************************//**
	 * @brief	Safely casts an object to the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @exception	axis::foundation::InvalidCastException	Thrown when
	 * 														the object
	 * 														cannot be
	 * 														casted to the
	 * 														specified
	 * 														type.
	 *
	 * @param [in,out]	obj	Pointer to the object to cast.
	 *
	 * @return	A pointer to the object.
	 **************************************************************************************************/
	template<class T> typename T * safe_cast(axis::foundation::assertion::Ascertainable * obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			throw axis::foundation::InvalidCastException(_T("Cannot cast to an object of type ") + typename T::GetClassName());
		}
		return static_cast<typename T *>(obj);
	}

	/**********************************************************************************************//**
	 * @brief	Casts, if possible, the specified object to a pointer of
	 * 			the specified type.
	 *
	 * @author	Renato T. Yamassaki
	 * @date	26 jun 2012
	 *
	 * @param	obj	Pointer to the object to cast.
	 *
	 * @return	A pointer to the object or null if it fails.
	 **************************************************************************************************/
	template<class T> typename T * const cast_or_null(const axis::foundation::assertion::Ascertainable * const obj)
	{
		if (!is_type_of<typename T>(obj))
		{
			return NULL;
		}
		return static_cast<typename T * const>(obj);
	}
}
