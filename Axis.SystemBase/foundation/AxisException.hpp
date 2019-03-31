/// <summary>
/// Contains definition for the axis::foundation::AxisException class.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once

#include "Axis.SystemBase.hpp"
#include "AxisString.hpp"
#include "SourceTraceHint.hpp"
#include "SourceHintCollection.hpp"

/// <summary>
/// Contains every classes, types and other symbols used by the axis application.
/// </summary>
namespace axis
{
	/// <summary>
	/// Contains classes and types that provides basic functionalities that aid in
	/// the tasks executed by the axis application.
	/// </summary>
	namespace foundation
	{
		/// <summary>
		/// Describes an exception thrown by the axis program or one of its components.
		/// </summary>
		class AXISSYSTEMBASE_API AxisException
		{
		private:
			String _description;
			AxisException *_innerException;

			SourceHintCollection &_traceHints;
		protected:
			virtual void Copy(const AxisException& e);
			void PushHints(AxisException& e) const;
		public:
			/// <summary>
			/// Creates a new exception with no description message.
			/// </summary>
			AxisException(void);

			/// <summary>
			/// Creates a new exception with a descriptive message.
			/// </summary>
			/// <param name="message">Reference to the message string to be attached to the exception.</param>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			AxisException(const String& message);

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
			AxisException(const String& message, const AxisException *innerException);

			/// <summary>
			/// Creates a new exception copied from an object.
			/// </summary>
			AxisException(const AxisException& ex);

			/// <summary>
			/// Creates a new exception with no descriptive message and chains it to another exception.
			/// </summary>
			/// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
			/// <remarks>
			/// The inner exception will be destroyed when this object is destroyed, so it is not recommended
			/// to any other part of the application handle the destruction process of the inner exception object.
			/// </remarks>
			AxisException(const AxisException *innerException);

			/// <summary>
			/// Releases all resources used by this object.
			/// </summary>
			virtual ~AxisException(void);

			/// <summary>
			/// Returns a reference to the inner exception object that caused this exception to be thrown.
			/// </summary>
			AxisException *GetInnerException(void) const;

			/// <summary>
			/// Sets the inner exception which caused this exception to be thrown.
			/// </summary>
			void SetInnerException(const AxisException *innerException);

			/// <summary>
			/// Returns the description message attached to this exception.
			/// </summary>
			String GetMessage(void) const;

			/// <summary>
			/// Sets the description message attached to this exception.
			/// </summary>
			/// <remarks>
			/// Once attached, the message string should not be used by other parts of the application
			/// since it will be handled and destroyed by the exception class itself.
			/// </remarks>
			void SetMessage(const String&  message);

			/// <summary>
			/// Copies the contents of an exception into this object.
			/// </summary>
			AxisException& operator =(const AxisException& ex);

			void AddTraceHint(const SourceTraceHint& hint);
			
			AxisException& operator << (const SourceTraceHint& hint);

			AxisException& operator << (const AxisException& innerException);

			bool HasSourceTraceHint(const SourceTraceHint& hint) const;

			/// <summary>
			/// Creates a copy of this exception.
			/// </summary>
			virtual AxisException& Clone(void) const;

			/**********************************************************************************************//**
			 * @fn	virtual String AxisException::GetTypeName(void) const;
			 *
			 * @brief	Gets the type name of this exception.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	08 mai 2012
			 *
			 * @return	The type name.
			 **************************************************************************************************/
			virtual String GetTypeName(void) const;
		};
	}
}