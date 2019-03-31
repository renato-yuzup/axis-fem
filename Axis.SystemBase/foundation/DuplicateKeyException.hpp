/// <summary>
/// Contains definitions for the class axis::foundation::DuplicateKeyException.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "AxisException.hpp"

namespace axis
{
  namespace foundation
  {
    /// <summary>
    /// Occurs when a collision with existing key in a collection is found.
    /// </summary>
    class AXISSYSTEMBASE_API DuplicateKeyException :
      public AxisException
    {
    public:
      /// <summary>
      /// Creates a new exception with no description message.
      /// </summary>	
      DuplicateKeyException(void);

      /// <summary>
      /// Creates a new exception pointing to where the configuration was not found.
      /// </summary>	
      DuplicateKeyException(String &location);

      /// <summary>
      /// Creates a new exception pointing to where the configuration was not found.
      /// </summary>	
      DuplicateKeyException(const char_type *location);

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
      DuplicateKeyException(const String& message, const AxisException * innerException);

      /// <summary>
      /// Creates a new exception with no descriptive message and chains it to another exception.
      /// </summary>	
      /// <param name="innerException">Another exception in chain that caused this exception to be thrown.</param>
      /// <remarks>
      /// The inner exception will be destroyed when this object is destroyed, so it is not recommended
      /// to any other part of the application handle the destruction process of the inner exception object.
      /// </remarks>
      DuplicateKeyException(const AxisException * innerException);

      DuplicateKeyException(const DuplicateKeyException& ex);

      DuplicateKeyException& operator =(const DuplicateKeyException& ex);

      /// <summary>
      /// Creates a copy of this exception.
      /// </summary>
      virtual AxisException& Clone(void) const;

      virtual String GetTypeName(void) const;
    };
  }
}
