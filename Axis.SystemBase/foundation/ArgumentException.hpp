/// <summary>
/// Contains definitions for the class axis::foundation::ArgumentException.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "AxisException.hpp"

namespace axis	
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when an argument passed to a function has an unexpected value.
		/// <summary>
		class AXISSYSTEMBASE_API ArgumentException : public AxisException
		{
		private:
			String _argumentName;
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			ArgumentException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="argumentName">Name of the argument which has an invalid value.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			ArgumentException(const String& argumentName);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="argumentName">Name of the argument which has an invalid value.</param>
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			ArgumentException(const String& argumentName, String& message);

			/// <summary>
			/// Creates a new exception with a descriptive message and links it to another exception.
			/// </summary>	
			/// <param name="argumentName">Name of the argument which has an invalid value.</param>
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself. 
			/// Also, the inner exception is destroyed when this object is destroyed.
			/// </remarks>
			ArgumentException(const String& argumentName, String& message, const AxisException * innerException);

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
			ArgumentException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			ArgumentException(const AxisException * innerException);

			/// <summary>
			/// Returns the name of the argument which caused this exception.
			/// </summary>	
			const String& GetArgumentName(void) const;

			ArgumentException(const ArgumentException& ex);

			ArgumentException& operator =(const ArgumentException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};

	}
}
