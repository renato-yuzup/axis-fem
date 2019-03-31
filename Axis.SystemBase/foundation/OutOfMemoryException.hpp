/// <summary>
/// Contains definitions for the class axis::foundation::OutOfMemoryException.
/// </summary>
/// <author>Renato T. Yamassaki</author>


#pragma once

#include "ApplicationErrorException.hpp"

namespace axis
{
	namespace foundation
	{
		
		/// <summary>
		/// Exception thrown when no memory can be allocated for an operation.
		/// </summary>	
		class AXISSYSTEMBASE_API OutOfMemoryException : public ApplicationErrorException
		{
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			OutOfMemoryException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			OutOfMemoryException(const String& message);

			/// <summary>
			/// Creates a new exception with a descriptive message and links it to another exception.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself. 
			/// Also, the inner exception is destroyed when this object is destroyed.
			/// </remarks>
			OutOfMemoryException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			OutOfMemoryException(const AxisException * innerException);

			OutOfMemoryException(const OutOfMemoryException& ex);

			OutOfMemoryException& operator =(const OutOfMemoryException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};

	}
}
