#ifndef __CUSTOMPARSERERROREXCEPTION_HPP
#define __CUSTOMPARSERERROREXCEPTION_HPP

#include "AxisParserException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when the parser of one of its components has reached the end
		/// of the input stream and some code blocks delimiters couldn't be found.
		/// </summary>
		class AXISSYSTEMBASE_API CustomParserErrorException :
			public AxisParserException
		{
		private:
			int _errorCode;
			axis::String _blockName;
		public:
			axis::String GetBlockName(void) const;
			int GetCustomErrorCode(void) const;
			axis::foundation::CustomParserErrorException& SetBlockName(const axis::String& blockName);
			axis::foundation::CustomParserErrorException& SetCustomErrorCode(int errorCode);


			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			CustomParserErrorException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			CustomParserErrorException(const String& message);

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
			CustomParserErrorException(const String& message, const AxisException * innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			CustomParserErrorException(const AxisException * innerException);

			CustomParserErrorException(const CustomParserErrorException& ex);

			CustomParserErrorException& operator =(const CustomParserErrorException& ex);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};


	}
}
#endif