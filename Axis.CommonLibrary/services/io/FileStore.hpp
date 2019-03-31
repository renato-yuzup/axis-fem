#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"
#include "StreamWriter.hpp"
#include "StreamReader.hpp"

namespace axis
{
	namespace services
	{
		namespace io
		{
			/**********************************************************************************************//**
			 * @class	FileStore
			 *
			 * @brief	Ensures that unique requests to files in the underlying file system.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	29 mai 2012
			 **************************************************************************************************/
			class AXISCOMMONLIBRARY_API FileStore
			{
			public:
				FileStore(const axis::String& basePath);

				/**********************************************************************************************//**
				 * @fn	FileStore::~FileStore(void);
				 *
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 **************************************************************************************************/
				~FileStore(void);

				/**********************************************************************************************//**
				 * @fn	virtual void FileStore::Destroy(void) const = 0;
				 *
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 **************************************************************************************************/
				void Destroy(void) const;

				/**********************************************************************************************//**
				 * @fn	size_type FileStore::BorrowedCount(void) const;
				 *
				 * @brief	Returns the number of opened files by this pool.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @return	.
				 **************************************************************************************************/
				size_type BorrowedCount(void) const;

				/**********************************************************************************************//**
				 * @fn	axis::services::io::StreamWriter& FileStore::AcquireStream(const axis::String& fileName);
				 *
				 * @brief	Borrows a stream with access to the specified file. This
				 * 			file is marked as unavailable until ReleaseStream is
				 * 			called with the appropriate set of parameters. If the
				 * 			requested file it is already unavailable, an exception is
				 * 			thrown.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param	fileName	File location.
				 *
				 * @return	A stream with access to the specified file.
				 **************************************************************************************************/
				axis::services::io::StreamWriter& AcquireStream(const axis::String& fileName);

				/**********************************************************************************************//**
				 * @fn	axis::services::io::StreamWriter& FileStore::AcquireStream(const axis::String& fileName,
				 * 		const axis::String& basePath);
				 *
				 * @brief	Borrows a stream with access to the specified file. This
				 * 			file is marked as unavailable until ReleaseStream is
				 * 			called with the appropriate set of parameters. If the
				 * 			requested file it is already unavailable, an exception is
				 * 			thrown.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param	fileName	File location.
				 * @param	basePath	Base path on which search for the file.
				 *
				 * @return	A stream with access to the specified file.
				 **************************************************************************************************/
				axis::services::io::StreamWriter& AcquireStream(const axis::String& fileName, const axis::String& basePath);

				/**********************************************************************************************//**
				 * <summary> Returns a previously acquired stream.</summary>
				 *
				 * <param name="fileName"> File location.</param>
				 *
				 * <returns> A stream with access to the specified file.</returns>
				 **************************************************************************************************/
				axis::services::io::StreamWriter& GetStream(const axis::String& fileName);

				/**********************************************************************************************//**
				 * <summary> Returns a previously acquired stream.</summary>
				 *
				 * <param name="fileName"> Filename of the file.</param>
				 * <param name="basePath"> Base search path of the file.</param>
				 *
				 * <returns> A stream with access to the specified file.</returns>
				 **************************************************************************************************/
				axis::services::io::StreamWriter& GetStream(const axis::String& fileName, const axis::String& basePath);

				/**********************************************************************************************//**
				 * <summary> Creates and returns a stream to a temporary file .</summary>
				 *
				 * <param name="fileName"> Filename of the temporary file.</param>
				 *
				 * <returns> A stream with access to the specified file.</returns>
				 **************************************************************************************************/
        axis::services::io::StreamWriter& CreateTempStream(const axis::String& fileName);

				/**********************************************************************************************//**
				 * <summary> Opens and returns a stream to a temporary file .</summary>
				 *
				 * <param name="fileName"> Filename of the temporary file.</param>
				 *
				 * <returns> A stream with access to the specified file.</returns>
				 **************************************************************************************************/
        axis::services::io::StreamReader& OpenTempStream(const axis::String& fileName);

				/**********************************************************************************************//**
				 * <summary> Closes and erases all temp streams created.</summary>
				 **************************************************************************************************/
        void ReleaseTempStreams(void);

				/**********************************************************************************************//**
				 * @fn	virtual void FileStore::PrepareAll(void) = 0;
				 *
				 * @brief	Do all the necessary steps in order to every stream acquired be writable.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 **************************************************************************************************/
				void PrepareAll(void);

				/**********************************************************************************************//**
				 * @fn	virtual void FileStore::ReleaseStream(axis::services::io::StreamWriter& writer) = 0;
				 *
				 * @brief	Releases the stream described by writer. The underlying
				 * 			resources are freed automatically. The corresponding file
				 * 			is also marked as available again.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param [in,out]	writer	The writer to release.
				 **************************************************************************************************/
				void ReleaseStream(axis::services::io::StreamWriter& writer);

				/**********************************************************************************************//**
				 * @fn	void FileStore::ReleaseStream(const axis::String& fileName);
				 *
				 * @brief	Releases the stream described by writer. The underlying
				 * 			resources are freed automatically. The corresponding file
				 * 			is also marked as available again.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param	fileName	File location.
				 **************************************************************************************************/
				void ReleaseStream(const axis::String& fileName);

				/**********************************************************************************************//**
				 * @fn	void FileStore::ReleaseStream(const axis::String& fileName,
				 * 		const axis::String& basePath);
				 *
				 * @brief	Releases the stream described by writer. The underlying
				 * 			resources are freed automatically. The corresponding file
				 * 			is also marked as available again.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param	fileName	File location.
				 * @param	basePath	Base path on which search for the file.
				 **************************************************************************************************/
				void ReleaseStream(const axis::String& fileName, const axis::String& basePath);

				/**********************************************************************************************//**
				 * @fn	void FileStore::ReleaseAll(void);
				 *
				 * @brief	Releases all opened streams and marks all opened files as
				 * 			available. Caution should be taken when using this method
				 * 			in order to avoid closing streams in use.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 **************************************************************************************************/
				void ReleaseAll(void);

				/**********************************************************************************************//**
				 * @fn	bool FileStore::IsAvailable(const axis::String& fileName) const;
				 *
				 * @brief	Query if the specified filename is already in use by any
				 * 			opened stream registered in this pool. Path is first
				 * 			converted to its canonical form and resolved in
				 * 			filesystem level to check if it points to the same file.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param	fileName	Filename to check.
				 *
				 * @return	true if available, false if not.
				 **************************************************************************************************/
				bool IsAvailable(const axis::String& fileName) const;

				/**********************************************************************************************//**
				 * @fn	bool FileStore::IsAvailable(const axis::String& fileName,
				 * 		const axis::String& basePath) const;
				 *
				 * @brief	Query if the specified filename is already in use by any
				 * 			opened stream registered in this pool. Path is first
				 * 			converted to its canonical form and resolved in
				 * 			filesystem level to check if it points to the same file.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @param	fileName	Filename to check.
				 * @param	basePath	Base path on which search for the file.
				 *
				 * @return	true if available, false if not.
				 **************************************************************************************************/
				bool IsAvailable(const axis::String& fileName, const axis::String& basePath) const;

				/**********************************************************************************************//**
				 * @fn	axis::String FileStore::GetBaseSearchPath(void) const;
				 *
				 * @brief	Returns the default base search location where this pool
				 * 			looks for a file or an empty string of not otherwise
				 * 			specified.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	29 mai 2012
				 *
				 * @return	The base search path.
				 **************************************************************************************************/
				axis::String GetBaseSearchPath(void) const;
      private:
        class FileStorePimpl;
        FileStorePimpl *pimpl_;
			};
		}
	}
}

