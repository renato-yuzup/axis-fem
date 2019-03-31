#ifndef __LEVEL2PARSEREXCEPTION_HPP
#define __LEVEL2PARSEREXCEPTION_HPP

#include "AxisParserException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when the parser of one of its components has reached the end
		/// of the input stream and some code blocks delimiters couldn't be found.
		/// </summary>
		class AXISSYSTEMBASE_API Level2ParserException :
			public AxisParserException
		{
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			Level2ParserException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			Level2ParserException(const String& message);

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
			Level2ParserException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			Level2ParserException(const AxisException * innerException);

			Level2ParserException(const Level2ParserException& ex);

			Level2ParserException operator =(const Level2ParserException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};


	}
}
#endif