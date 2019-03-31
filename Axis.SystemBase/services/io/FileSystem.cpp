#include "FileSystem.hpp"

#include <shlwapi.h>
#include <string>

#include "foundation/Settings/SystemSettings.hpp"
#include "boost/filesystem.hpp"
#include "foundation/IOException.hpp"

// we have to undefine this macro if it exists (Windows) in order 
// to remove any conflict
#ifdef GetFileTitle
	#undef GetFileTitle
#endif


// After version 1.49 of Boost, the filesystem3 is deprecated
// in favor of the filesystem namespace.
#if BOOST_VERSION > 104900
	#define AXIS_BOOST_FILESYSTEM	boost::filesystem
#else
	#define AXIS_BOOST_FILESYSTEM	boost::filesystem3
#endif

#ifdef _UNICODE
	#define GETCWD _wgetcwd
	typedef boost::filesystem::wpath boostpath;
	typedef std::wstring std_string;
#else
	#define GETCWD _getcwd
	typedef boost::filesystem::path boostpath;
	typedef std::string std_string;
#endif

using namespace axis::services::io;
using namespace axis::foundation;

FileSystem::FileSystem(void)
{
	// no-op
}

FileSystem::~FileSystem(void)
{
	// no-op
}

axis::String FileSystem::GetDefaultConfigurationFileLocation( void )
{
	String sAppPath = GetApplicationFolder();
	char_type *buf = (char_type *)sAppPath.data();
	boostpath configFile(buf);
	String configFileName = axis::foundation::settings::SystemSettings::DefaultConfigurationFile;

	// CHAR *bufConfigPath = configFile.append(configFileName.begin(),configFileName.end()).filename();
	const String sConfigPath = ConcatenatePath(sAppPath, axis::foundation::settings::SystemSettings::DefaultConfigurationFile);

	return sConfigPath;
}

axis::String axis::services::io::FileSystem::GetLocaleFolder( void )
{
	String appFolder = GetApplicationFolder();
	String localeFolder = axis::foundation::settings::SystemSettings::LocaleFolderName;
	return ConcatenatePath(appFolder, localeFolder);
}

axis::String FileSystem::GetApplicationFolder( void )
{
	return AXIS_BOOST_FILESYSTEM::current_path().make_preferred().c_str();
}

bool axis::services::io::FileSystem::IsAbsolutePath( const axis::String& pathString )
{
	boostpath p(pathString.data());
	return p.is_absolute() && p.has_parent_path();
}

axis::String axis::services::io::FileSystem::ConcatenatePath( const axis::String& basePath, const axis::String& appendPath )
{
	boostpath p1(basePath.data()), p2(appendPath.data());
	return axis::String((p1 / p2).make_preferred().c_str());
}

axis::String axis::services::io::FileSystem::ToCanonicalPath( const axis::String& relativePath )
{
 	boostpath p = boostpath(relativePath.data()).make_preferred();
	try
	{
		if (exists(p))
		{	// if path exists, it is better to use Boost Library
			boostpath absolutePath = AXIS_BOOST_FILESYSTEM::canonical(p).make_preferred();
			return absolutePath.c_str();
		}
	}
	catch (...)
	{	// file might have been deleted since we checked for its existence; fallback to Windows API		
	}

	// use Windows API
	boostpath absolutePath = AXIS_BOOST_FILESYSTEM::absolute(p);
	char_type *buf = new char_type[MAX_PATH];

	BOOL result = PathCanonicalize(buf, absolutePath.c_str());
	boostpath aux(buf);
	delete [] buf;
	return aux.make_preferred().c_str();
}

bool axis::services::io::FileSystem::IsDirectory( const axis::String& path )
{
	boostpath p(path.data());
	return AXIS_BOOST_FILESYSTEM::is_directory(p);
}

bool axis::services::io::FileSystem::IsDirectoryEmpty( const axis::String& path )
{
	boostpath p(path.data());
	return AXIS_BOOST_FILESYSTEM::is_empty(p);
}

bool axis::services::io::FileSystem::ExistsFile( const axis::String& path )
{
	boostpath p(path.data());
	return AXIS_BOOST_FILESYSTEM::exists(p);
}

bool axis::services::io::FileSystem::IsSameFile( const axis::String& path1, const axis::String& path2 )
{
	boostpath p1(path1.data()), p2(path2.data());
	if (AXIS_BOOST_FILESYSTEM::exists(p1) || AXIS_BOOST_FILESYSTEM::exists(p2))
	{
		return AXIS_BOOST_FILESYSTEM::equivalent(p1, p2);
	}
	else
	{
		return p1.compare(p2) == 0;
	}
}

axis::String axis::services::io::FileSystem::GetFilePath( const axis::String& filename )
{
	boostpath p(filename);
	return p.parent_path().make_preferred().c_str();
}

axis::String axis::services::io::FileSystem::GetFileTitle( const axis::String& filename )
{
	boostpath p(filename);
	return p.stem().make_preferred().c_str();
}

axis::String axis::services::io::FileSystem::GetFileExtension( const axis::String& filename )
{
	boostpath p(filename);
	return p.extension().make_preferred().c_str();
}

axis::String axis::services::io::FileSystem::RemoveFileExtension( const axis::String& filename )
{
	boostpath path(filename);
	boostpath root = path.parent_path();
	boostpath filetitle = path.stem();
	return (root / filetitle).make_preferred().c_str();
}

axis::String axis::services::io::FileSystem::ReplaceFileExtension( const axis::String& filename, const axis::String& newExtension )
{
	boostpath path(filename);
	boostpath root = path.parent_path();
	boostpath filetitle = path.stem();
	boostpath extension(newExtension.data());
	boostpath finalPath((String((root / filetitle).c_str()) + _T(".") + newExtension).c_str());
	return finalPath.make_preferred().c_str();
}

axis::String axis::services::io::FileSystem::GetLibraryFileExtension( void )
{
	return _T("dll");
}

void axis::services::io::FileSystem::PurgeFile( const axis::String& filename )
{
  boostpath path(filename);
  try
  {
    if (!boost::filesystem::remove(path))
    {
      throw axis::foundation::IOException(_T("File not found."));
    }
  }
  catch (...)
  {
    throw axis::foundation::IOException(_T("Could not delete file '") + filename + _T("'."));
  }
}
