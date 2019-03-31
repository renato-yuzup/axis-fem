#pragma once

#if defined(AXISPHYSALIS_EXPORTS)
#define AXISPHYSALIS_API  __declspec(dllexport)
#else
#define AXISPHYSALIS_API  __declspec(dllimport)
#endif
