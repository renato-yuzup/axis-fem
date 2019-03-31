/// <summary>
/// Contains definition for the class axis::foundation::IOException.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "AxisException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when a syntax error is found while reading the application's configuration file.
		/// </summary>
		class AXISSYSTEMBASE_API SyntaxMismatchException : public AxisException
		{
		private:
			int _lineNumber;
			String _fileName;
			String _reason;
		protected:
			virtual void Copy(const AxisException& e);
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			SyntaxMismatchException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			SyntaxMismatchException(const String& message);

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
			SyntaxMismatchException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			SyntaxMismatchException(const AxisException * innerException);

			/// <summary>
			/// Returns the line number where the error was detected.
			/// </summary>	
			int GetLine(void);

			/// <summary>
			/// Sets the line number where the error was detected.
			/// </summary>	
			/// <param name="value">Line number.</param>
			void SetLine(int value);

			/// <summary>
			/// Returns the filename where the error was detected.
			/// </summary>	
			const String GetFileName(void);

			/// <summary>
			/// Sets the filename where the error was detected.
			/// </summary>	
			/// <param name="filename">The absolute file path.</param>
			void SetFileName(const String filename);

			/// <summary>
			/// Sets the filename where the error was detected.
			/// </summary>	
			/// <param name="filename">The absolute file path.</param>
			void SetFileName(const char_type *filename);
			
			/// <summary>
			/// Returns the message explaining the reason for the error.
			/// </summary>	
			const String GetMessageReason(void);

			/// <summary>
			/// Sets the message explaining the reason for the error.
			/// </summary>	
			/// <param name="message">The message reason.</param>
			void SetMessageReason(const String message);

			/// <summary>
			/// Sets the message explaining the reason for the error.
			/// </summary>	
			/// <param name="message">The message reason.</param>
			void SetMessageReason(const char_type *message);

			SyntaxMismatchException(const SyntaxMismatchException& ex);

			SyntaxMismatchException& operator =(const SyntaxMismatchException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};
	}
}

