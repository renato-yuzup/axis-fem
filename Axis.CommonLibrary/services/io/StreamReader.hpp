#pragma once
#include "AxisString.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis
{
	namespace services
	{
		namespace io
		{
			/// <summary>
			/// Describes a class which is capable of reading data from a stream.
			/// </summary>
			class AXISCOMMONLIBRARY_API StreamReader
			{
			public:
				/// <summary>
				/// Reads until a non-blank line is found in the stream or EOF was reached.
				/// </summary>
				/// <param name="s">The string object where to allocate the line contents.</param>
				/// <exception cref="axis::foundation::IOException">An unexpected error occurred while trying to read the line from the stream.</exception>
				virtual void ReadLine(axis::String& s) = 0;

				/// <summary>
				/// Pushes back last line read to memory so that the next line which
				/// should be read remains the same.
				/// </summary>
				/// <exception cref="axis::foundation::InvalidOperationExceptopn">If no line was read until now.</exception>
				/// <exception cref="axis::foundation::NotSupportedException">If a second consecutively request is made before the pushed back line should be read.</exception>
				virtual void PushBackLine(void) = 0;

				/// <summary>
				/// Indicates if the end of the stream was reached.
				/// </summary>
				/// <returns>
				/// True if EOF was reached or False otherwise.
				/// </returns>
				/// <remarks>
				/// If the stream is not open, this method will always return False.
				/// </remarks>
				virtual bool IsEOF(void) const = 0;

				/// <summary>
				/// Indicates if the stream is still opened.
				/// </summary>
				/// <returns>
				/// True if it is opened, False otherwise.
				/// </returns>
				virtual bool IsOpen(void) const = 0;

				/// <summary>
				/// Closes the stream and releases associated resources.
				/// </summary>
				/// <exception cref="axis::foundation::IOException">An error occurred while trying to close the stream.</exception>
				virtual void Close(void) = 0;

				/// <summary>
				/// Returns a string which describes this stream.
				/// </summary>
				virtual axis::String GetStreamPath(void) const = 0;

				/// <summary>
				/// Returns the line number of the last line read from the stream.
				/// </summary>
				/// <returns>
				/// Returns the line number (starting at 1) of the last line read
				/// from the stream. If no line has been read, the value 0 (zero) is
				/// returned.
				/// </returns>
				virtual unsigned long GetLastLineNumber(void) const = 0;

				/// <summary>
				/// Closes current stream and reopen so that this object points to
				/// the start of the stream.
				/// </summary>
				virtual void Reset(void) = 0;

				/// <summary>
				/// Destroys this object.
				/// </summary>
				virtual ~StreamReader(void) { }	// no-op
			};
		}
	}
}


