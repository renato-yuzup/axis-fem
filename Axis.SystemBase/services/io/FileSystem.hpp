#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/ApplicationErrorException.hpp"

using namespace axis::foundation;

namespace axis
{
	namespace services
	{
		namespace io
		{
			/// <summary>
			/// Provides information about files and folders used by the application.
			/// This class is static and cannot be manually instantiated.
			/// </summary>
			/// <remarks>
			/// Multiple accesses (concurrent) to the unique instance of this object is permitted,
			/// however, it is recommended that the first access does not occur in a possible
			/// concurrency access, as it might return two or more different instances, with the memory
			/// pointer to one of them being discarded resulting in memory leak.
			/// </remarks>
			class AXISSYSTEMBASE_API FileSystem
			{
			private:
				/// <summary>
				/// Creates a new instance of this class.
				/// </summary>
				FileSystem(void);
			public:
				/// <summary>
				/// Destroys this object from the memory releasing all resources
				/// allocated for it.
				/// </summary>
				~FileSystem(void);

				/// <summary>
				/// Returns the path where the application files are located.
				/// </summary>
				static axis::String GetApplicationFolder(void);

				/// <summary>
				/// Returns the filename of the application configuration file.
				/// </summary>
				static axis::String GetDefaultConfigurationFileLocation(void);

				/// <summary>
				/// Returns the path where the application locale libraries are located.
				/// </summary>
				static axis::String GetLocaleFolder(void);

				static bool IsAbsolutePath(const axis::String& pathString);

				static axis::String ConcatenatePath(const axis::String& basePath, const axis::String& appendPath);

				static axis::String ToCanonicalPath(const axis::String& relativePath);

				static bool IsDirectory(const axis::String& path);

				static bool IsDirectoryEmpty(const axis::String& path);

				static bool ExistsFile(const axis::String& path);

				static bool IsSameFile(const axis::String& path1, const axis::String& path2);

				static axis::String GetFilePath(const axis::String& filename);
				static axis::String GetFileTitle(const axis::String& filename);
				static axis::String GetFileExtension(const axis::String& filename);
				static axis::String RemoveFileExtension(const axis::String& filename);
				static axis::String ReplaceFileExtension(const axis::String& filename, const axis::String& newExtension);

				static axis::String GetLibraryFileExtension(void);

        static void PurgeFile(const axis::String& filename);
			};
		}
	}
}
