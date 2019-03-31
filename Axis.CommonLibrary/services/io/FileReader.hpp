/// <summary>
/// Contains definition for the class axis::Service::io::FileReader.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once

#include <fstream>
#include "AxisString.hpp"
#include "StreamReader.hpp"

namespace axis
{

	namespace services
	{
		namespace io
		{
			/// <summary>
			/// Implements operations to read data from a file in disk.
			/// </summary>
			class AXISCOMMONLIBRARY_API FileReader : public StreamReader
			{
			private:
				class InternalFileData;
				InternalFileData *data_;
			public:
				/// <summary>
				/// Creates a new instance of this class and opens an existing file.
				/// </summary>
				/// <param name="filename">The absolute path pointing to the file.</param>
				/// <exception cref="axis::foundation::IOException">An error occurred while opnening the file.</exception>
				/// <exception cref="axis::foundation::OutOfMemoryException">Couldn't allocate sufficient memory for input buffer.</exception>
				FileReader(const axis::String& filename);

				/// <summary>
				/// Reads until a non-blank line is found in the stream or EOF was reached.
				/// </summary>
				/// <param name="s">The string object where to allocate the line contents.</param>
				/// <exception cref="axis::foundation::IOException">An unexpected error occurred while trying to read the line from the file.</exception>
				virtual void ReadLine(axis::String& s);

				/// <summary>
				/// Pushes back last line read to memory so that the next line which
				/// should be read remains the same.
				/// </summary>
				/// <exception cref="axis::foundation::InvalidOperationExceptopn">If no line was read until now.</exception>
				/// <exception cref="axis::foundation::NotSupportedException">If a second consecutively request is made before the pushed back line should be read.</exception>
				virtual void PushBackLine(void);

				/// <summary>
				/// Indicates if the end of the file was reached.
				/// </summary>
				/// <returns>
				/// True if EOF was reached or False otherwise.
				/// </returns>
				/// <remarks>
				/// If the file is not open, this method will always return False.
				/// </remarks>
				virtual bool IsEOF(void) const;

				/// <summary>
				/// Indicates if the file is still opened.
				/// </summary>
				/// <returns>
				/// True if the file is opened, False otherwise.
				/// </returns>
				virtual bool IsOpen(void) const;

				/// <summary>
				/// Closes the file and releases associated resources.
				/// </summary>
				/// <exception cref="axis::foundation::IOException">An error occurred while trying to close the file.</exception>
				virtual void Close(void);

				/// <summary>
				/// Returns a string which describes this stream.
				/// </summary>
				virtual axis::String GetStreamPath(void) const;
				
				/// <summary>
				/// Returns the line number of the last line read from the stream.
				/// </summary>
				/// <returns>
				/// Returns the line number (starting at 1) of the last line read
				/// from the stream. If no line has been read, the value 0 (zero) is
				/// returned.
				/// </returns>
				virtual unsigned long GetLastLineNumber(void) const;

				/// <summary>
				/// Closes current stream and reopen so that this object points to
				/// the start of the stream.
				/// </summary>
				virtual void Reset(void);				

				/// <summary>
				/// Destroys this object.
				/// </summary>
				/// <remarks>
				/// The <see cref="Close" /> method is automatically called prior to object destruction if it hasn't been done so.
				/// </remarks>
				virtual ~FileReader(void);
			};
		}
	}
}

