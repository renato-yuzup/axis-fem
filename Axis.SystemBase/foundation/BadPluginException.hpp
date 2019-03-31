/// <summary>
/// Contains definitions for the class axis::foundation::BadPluginException.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "AxisException.hpp"

namespace axis	
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when a valid plugin failed to load.
		/// <summary>
		class AXISSYSTEMBASE_API BadPluginException : public AxisException
		{
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			BadPluginException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			BadPluginException(const String& message);

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
			BadPluginException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			BadPluginException(const AxisException * innerException);

			BadPluginException(const BadPluginException& ex);

			BadPluginException& operator =(const BadPluginException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};

	}
}
