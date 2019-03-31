/// <summary>
/// Contains definitions for the class axis::foundation::ConfigurationNotFoundException.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "AxisException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when an expected setting was not found in the application's configuration file.
		/// </summary>
		class AXISSYSTEMBASE_API ConfigurationNotFoundException :
			public AxisException
		{
		private:
			String _path;
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			ConfigurationNotFoundException(void);

			/// <summary>
			/// Creates a new exception pointing to where the configuration was not found.
			/// </summary>	
			ConfigurationNotFoundException(String &location);

			/// <summary>
			/// Creates a new exception pointing to where the configuration was not found.
			/// </summary>	
			ConfigurationNotFoundException(const char_type *location);

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
			ConfigurationNotFoundException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			ConfigurationNotFoundException(const AxisException * innerException);

			/// <summary>
			/// Returns the path where the configuration was not found.
			/// </summary>	
			String GetPath(void);

			/// <summary>
			/// Sets the path where the configuration was not found.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			void SetPath(const String & s);

			ConfigurationNotFoundException(const ConfigurationNotFoundException& ex);

			ConfigurationNotFoundException& operator =(const ConfigurationNotFoundException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};
	}
}
