/// <summary>
/// Contains definitions for the class axis::foundation::AssertionFailedException.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "ApplicationErrorException.hpp"

namespace axis	
{
	namespace foundation
	{
		/**********************************************************************************************//**
		 * @brief	Exception for signalling assertion failed errors.
		 *
		 * @author	Renato T. Yamassaki
		 * @date	20 ago 2012
		 *
		 * @sa	ApplicationErrorException
		 **************************************************************************************************/
		class AXISSYSTEMBASE_API AssertionFailedException : public ApplicationErrorException
		{
		public:

			/**********************************************************************************************//**
			 * @brief	Creates a new exception with no description message.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 **************************************************************************************************/
			AssertionFailedException(void);

			/**********************************************************************************************//**
			 * @brief	Creates a new exception with a descriptive message.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 *
			 * @param	message	Reference to the message string to be attached to
			 * 					the exception.
			 *
			 * ### remarks	Once attached, the message string should not be used
			 * 				by other parts of the application since it will be
			 * 				handled and destroyed by the exception class itself.
			 **************************************************************************************************/
			AssertionFailedException(const String& message);

			/**********************************************************************************************//**
			 * @brief	Creates a new exception with a descriptive message.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 *
			 * @param	expression  	String representing the evaluated expression.
			 * @param	functionName	Function name where the assertion occurred.
			 * @param	filename		Filename for the corresponding source file.
			 * @param	lineNumber  	The line number of the instruction.
			 **************************************************************************************************/
			AssertionFailedException(const AsciiString& expression, const AsciiString& functionName, const AsciiString& filename, long lineNumber);

			/**********************************************************************************************//**
			 * @brief	Creates a new exception with a descriptive message and
			 * 			links it to another exception.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 *
			 * @param	message		  	Reference to the message string to be
			 * 							attached to the exception.
			 * @param	innerException	Another exception in chain that caused
			 * 							this exception to be thrown.
			 *
			 * ### remarks	Once attached, the message string should not be used
			 * 				by other parts of the application since it will be
			 * 				handled and destroyed by the exception class itself.
			 * 				Also, the inner exception is destroyed when this
			 * 				object is destroyed.
			 **************************************************************************************************/
			AssertionFailedException(const String& message, const AxisException * innerException);

			/**********************************************************************************************//**
			 * @brief	Creates a new exception with no descriptive message and
			 * 			chains it to another exception.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 *
			 * @param	innerException	Another exception in chain that caused
			 * 							this exception to be thrown.
			 *
			 * ### remarks	The inner exception will be destroyed when this
			 * 				object is destroyed, so it is not recommended to any
			 * 				other part of the application handle the destruction
			 * 				process of the inner exception object.
			 **************************************************************************************************/
			AssertionFailedException(const AxisException * innerException);

			/**********************************************************************************************//**
			 * @brief	Copy constructor.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 *
			 * @param	ex	The ex.
			 **************************************************************************************************/
			AssertionFailedException(const AssertionFailedException& ex);

			AssertionFailedException& operator =(const AssertionFailedException& ex);

			/**********************************************************************************************//**
			 * @brief	Creates a copy of this exception.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	20 ago 2012
			 *
			 * @return	A copy of this object.
			 **************************************************************************************************/
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};

	}
}
