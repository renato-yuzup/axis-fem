#pragma once

#if defined(AXISCORE_EXPORTS)
#define AXISCORE_API __declspec(dllexport)
#else
#define AXISCORE_API __declspec(dllimport)
#endif
