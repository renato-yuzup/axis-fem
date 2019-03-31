#ifndef __SYMBOLREDEFINEDEXCEPTION_HPP
#define __SYMBOLREDEFINEDEXCEPTION_HPP

#include "PreProcessorException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when an invalid directive is found in the input stream.
		/// </summary>
		class AXISSYSTEMBASE_API SymbolRedefinedException :
			public PreProcessorException
		{
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			SymbolRedefinedException(void);

			/// <summary>
			/// Creates a new exception describing the location of the error
			/// </summary>	
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			/// <param name="column">The column index on which the error occurred. A non-positive number indicates an undefined column.</param>
			SymbolRedefinedException(const String& filename, unsigned long lineNumber, long column);

			/// <summary>
			/// Creates a new exception describing the location of the error
			/// </summary>	
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			SymbolRedefinedException(const String& filename, unsigned long lineNumber);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			/// <param name="column">The column index on which the error occurred. A non-positive number indicates an undefined column.</param>
			SymbolRedefinedException(const String& message, String& filename, unsigned long lineNumber, long column);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			SymbolRedefinedException(const String& message);

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
			SymbolRedefinedException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			SymbolRedefinedException(const AxisException * innerException);

			/// <summary>
			/// Creates a new exception and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			/// <param name="column">The column index on which the error occurred. A non-positive number indicates an undefined column.</param>
			SymbolRedefinedException(const AxisException *innerException, const String& filename, unsigned long lineNumber, long column);

			/// <summary>
			/// Creates a new exception and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			SymbolRedefinedException(const AxisException *innerException, const String& filename, unsigned long lineNumber);

			SymbolRedefinedException(const SymbolRedefinedException& ex);

			SymbolRedefinedException& operator =(const SymbolRedefinedException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};


	}
}
#endif