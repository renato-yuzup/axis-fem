#pragma once

#if defined(AXISCAPSICUM_EXPORTS)
#define AXISCAPSICUM_API __declspec(dllexport)
#else
#define AXISCAPSICUM_API __declspec(dllimport)
#endif
