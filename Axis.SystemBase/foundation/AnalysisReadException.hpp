#ifndef __INVALIDPREPROCESSORDIRECTIVEEXCEPTION_HPP
#define __INVALIDPREPROCESSORDIRECTIVEEXCEPTION_HPP

#include "PreProcessorException.hpp"

namespace axis
{
	namespace foundation
	{
		/// <summary>
		/// Occurs when an invalid directive is found in the input stream.
		/// </summary>
		class AXISSYSTEMBASE_API AnalysisReadException :
			public AxisException
		{
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>	
			AnalysisReadException(void);


			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>	
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			AnalysisReadException(const String& message);

			/// <summary>
			/// Creates a new exception by copying another exception.
			/// </summary>	
			/// <param name="exception">The other exception.</param>
			AnalysisReadException(const AnalysisReadException& exception);

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
			AnalysisReadException(const String& message, const AxisException *innerException);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>	
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			AnalysisReadException(const AxisException *innerException);

			AnalysisReadException& operator =(const AnalysisReadException& e);

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			virtual String GetTypeName(void) const;
		};


	}
}
#endif