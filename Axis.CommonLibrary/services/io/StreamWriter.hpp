#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace io
		{
			class AXISCOMMONLIBRARY_API StreamWriter
			{
			public:
				enum WriteMode
				{
					kOverwrite,
					kAppend				
				};

				enum LockMode
				{
					kExclusiveMode,
					kSharedMode
				};

				/**********************************************************************************************//**
				 * @fn	virtual StreamWriter::~StreamWriter(void);
				 *
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 **************************************************************************************************/
				virtual ~StreamWriter(void);

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::WriteLine(const axis::String& s) = 0;
				 *
				 * @brief	Write the string and append a end-of-line sequence to the stream.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @param	s	The string to write.
				 *
				 * ### exception	axis::foundation::IOException	An unexpected
				 * 													error occurred
				 * 													while trying to
				 * 													write the line
				 * 													to the stream.
				 **************************************************************************************************/
				virtual void WriteLine(const axis::String& s) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::WriteLine(void) = 0;
				 *
				 * @brief	Write the string and append a end-of-line sequence to the
				 * 			stream.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 **************************************************************************************************/
				virtual void WriteLine(void) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::WriteLine(const axis::String& s) = 0;
				 *
				 * @brief	Write the string to the stream.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @param	s	The string to write.
				 *
				 * ### exception	axis::foundation::IOException	An unexpected
				 * 													error occurred
				 * 													while trying to
				 * 													write the line
				 * 													to the stream.
				 **************************************************************************************************/
				virtual void Write(const axis::String& s) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual axis::String StreamWriter::GetEndOfLineSequence(void) const = 0;
				 *
				 * @brief	Gets the end of line sequence used by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	The end of line sequence.
				 **************************************************************************************************/
				virtual axis::String GetEndOfLineSequence(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::SetEndOfLineSequence(const axis::String& eol) = 0;
				 *
				 * @brief	Sets the end of line sequence to be used in subsequent
				 * 			write calls of this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @param	eol	The end-of-line sequence to use.
				 **************************************************************************************************/
				virtual void SetEndOfLineSequence(const axis::String& eol) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual unsigned long StreamWriter::GetBytesWritten(void) const = 0;
				 *
				 * @brief	Gets the number of bytes written so far by this object to the stream. It doesn't 
				 * 			necessarily reflect the total stream size.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	The bytes written.
				 **************************************************************************************************/
				virtual unsigned long GetBytesWritten(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual bool StreamWriter::IsAutoFlush(void) const = 0;
				 *
				 * @brief	Query if this object has automatic flush capability.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	true if it is automatic flush capable, false if not.
				 **************************************************************************************************/
				virtual bool IsAutoFlush(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual bool StreamWriter::IsBuffered(void) const = 0;
				 *
				 * @brief	Query if this object is buffered.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	true if buffered, false if not.
				 **************************************************************************************************/
				virtual bool IsBuffered(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual unsigned long StreamWriter::GetBufferSize(void) const = 0;
				 *
				 * @brief	Gets the total buffer size (free and used space) in memory.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	The buffer size.
				 **************************************************************************************************/
				virtual unsigned long GetBufferSize(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual unsigned long StreamWriter::GetBufferUsedSpace(void) const = 0;
				 *
				 * @brief	Gets the buffer used space.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	The buffer used space.
				 **************************************************************************************************/
				virtual unsigned long GetBufferUsedSpace(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual bool StreamWriter::IsOpen(void) const = 0;
				 *
				 * @brief	Indicates if the stream is still opened.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	True if it is opened, False otherwise.
				 **************************************************************************************************/
				virtual bool IsOpen(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::Open(WriteMode writeMode = WriteMode::kOverwrite,
				 * 		LockMode lockMode = LockMode::kSharedMode) = 0;
				 *
				 * @brief	Opens the stream for write mode.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @param	writeMode	(optional) the write mode.
				 * @param	lockMode 	(optional) the lock mode.
				 **************************************************************************************************/
				virtual void Open(WriteMode writeMode = kOverwrite, LockMode lockMode = kSharedMode) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::Flush(void) = 0;
				 *
				 * @brief	Request this object to write the entire contents of its buffer to the underlined stream.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 **************************************************************************************************/
				virtual void Flush(void) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::ToggleFlush(void) = 0;
				 *
				 * @brief	Tells this object that if the buffer is full, the object has permission to flush 
				 * 			its buffer until the next write command.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 **************************************************************************************************/
				virtual void ToggleFlush(void) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::Close(void) = 0;
				 *
				 * @brief	Closes the stream and releases associated resources.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * ### exception	axis::foundation::IOException	An error occurred
				 * 													while trying to
				 * 													close the stream.
				 **************************************************************************************************/
				virtual void Close(void) = 0;

				/**********************************************************************************************//**
				 * @fn	virtual const axis::String& StreamWriter::GetStreamPath(void) const = 0;
				 *
				 * @brief	Returns a string which describes this stream.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 *
				 * @return	The description.
				 **************************************************************************************************/
				virtual axis::String GetStreamPath(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void StreamWriter::Destroy(void) const = 0;
				 *
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	03 mai 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;
			};
		
		}
	}
}

