#ifndef __AXISPARSEREXCEPTION_HPP
#define __AXISPARSEREXCEPTION_HPP
#include "Axis.SystemBase.hpp"
#include "AxisException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when an some form of syntax can't be interpreted in the input file.
		/// <summary>
		class AXISSYSTEMBASE_API AxisParserException : public AxisException
		{
		private:
			String _filename;
			unsigned long _lineNumber;
			long _column;
		protected:
			virtual void Copy(const AxisException& ex);
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			AxisParserException(void);

			/// <summary>
			/// Creates a new exception describing the location of the error
			/// </summary>	
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			/// <param name="column">The column index on which the error occurred. A non-positive number indicates an undefined column.</param>
			AxisParserException(const String& filename, unsigned long lineNumber, long column);

			/// <summary>
			/// Creates a new exception describing the location of the error
			/// </summary>	
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			AxisParserException(const String& filename, unsigned long lineNumber);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			/// <param name="column">The column index on which the error occurred. A non-positive number indicates an undefined column.</param>
			AxisParserException(const String& message, const String& filename, unsigned long lineNumber, long column);

			/// <summary>
			/// Creates a new exception and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			/// <param name="column">The column index on which the error occurred. A non-positive number indicates an undefined column.</param>
			AxisParserException(const AxisException *innerException, const String& filename, unsigned long lineNumber, long column);

			/// <summary>
			/// Creates a new exception and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <param name="filename">The file (or stream description) on which the error occurred.</param>
			/// <param name="lineNumber">The line index on which the error occurred.</param>
			AxisParserException(const AxisException *innerException, const String& filename, unsigned long lineNumber);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			AxisParserException(const String& message);

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
			AxisParserException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			AxisParserException(const AxisException * innerException);

			/// <summary>
			/// Returns if a valid column index was informed in this exception.
			/// </summary>
			bool IsColumnDefined(void) const;

			/// <summary>
			/// Returns the column index on which the syntax or lexical analysis stopped.
			/// </summary>
			/// <returns>
			/// Returns a positive number representing the column index or a 
			/// non-positive number if no column index was informed.
			/// </returns>
			long GetColumnIndex(void) const;

			/// <summary>
			/// Returns the line index on which the syntax or lexical analysis stopped.
			/// </summary>
			unsigned long GetLineIndex(void) const;

			/// <summary>
			/// Returns the filename (or a string describing a stream) where the
			/// syntactic (or lexical) analysis stopped.
			/// </summary>
			String GetFileName(void) const;

			AxisParserException(const AxisParserException& ex);

			AxisParserException& operator =(const AxisParserException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};
	}
}

#endif