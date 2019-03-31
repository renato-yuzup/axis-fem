/// <summary>
/// Contains all macros definitions used in this application. This file
// should not be used directly; instead the file stdafx.h should be used.
/// </summary>
#ifndef __MACROS_HPP
#define __MACROS_HPP

#include "AxisString.hpp"
#include <fstream>

/* According to property_tree class documentation, we have to defined a
   specialized type for the custom key type that we want to use.
*/
/*template <>
struct boost::property_tree::path_of< axis::String >
{
	typedef axis::String _string;
	typedef string_path< _string, id_translator<_string> > type;
};*/

#ifdef _UNICODE

    //typedef boost::property_tree::wiptree config_tree;
    //typedef boost::property_tree::wpath config_path;
    typedef std::wifstream input_stream;
    typedef	std::wfilebuf input_stream_buffer;
    typedef wchar_t string_unit;
	typedef std::wstring std_string;

    /* Function definitions */
    #define GETCWD _wgetcwd

    /* Constants */
    #define AXIS_CONFIG_FILE										L"axis.config"
    #define AXIS_CONFIG_ROOT										L"axisConfiguration"
    #define AXIS_CONFIG_DIRECTORIES_SECTION							L"axisConfiguration.directories"
    #define AXIS_CONFIG_DIRECTORY_ENTRY								L"path"
    #define AXIS_CONFIG_DIRECTORY_FEATURE_ATTRIBUTE					L"feature"
    #define AXIS_CONFIG_DIRECTORY_LOCATION_ATTRIBUTE				L"location"
    #ifdef _WINDOWS
        #define AXIS_ENDLINE										L"\n"
    #else
        #define AXIS_ENDLINE										L"\r"
    #endif
#else
    typedef	boost::filesystem::path boostpath;
    typedef boost::property_tree::iptree config_tree;
    typedef boost::property_tree::path config_path;
    typedef std::ifstream input_stream;
    typedef	std::filebuf input_stream_buffer;
    typedef char string_unit;
	typedef std::string std_string;

    /* Function definitions */
    #define GETCWD _getcwd

    /* Constants */
    #define AXIS_CONFIG_FILE										"axis.config"
    #define AXIS_CONFIG_ROOT										"axisConfiguration"
    #define AXIS_CONFIG_DIRECTORIES_SECTION							"axisConfiguration.directories"
    #define AXIS_CONFIG_DIRECTORY_ENTRY								"path"
    #define AXIS_CONFIG_DIRECTORY_FEATURE_ATTRIBUTE					"feature"
    #define AXIS_CONFIG_DIRECTORY_LOCATION_ATTRIBUTE				"location"
    #ifdef _WINDOWS
        #define AXIS_ENDLINE										"\n"
    #else
        #define AXIS_ENDLINE										"\r"
    #endif
#endif

typedef unsigned long big_number;

/* Program configuration macros */

// Macros related to file manipulation
#define AXIS_INPUTFILE_BUFFER_LENGTH								4096*1024	// 4 MB

// Macros related to preprocessor settings
#define AXIS_PREPROCESSOR_FILE_STACK_DEPTH							256

#endif
