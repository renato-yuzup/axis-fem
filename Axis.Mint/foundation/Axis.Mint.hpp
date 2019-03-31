#pragma once

#if defined(AXISMINT_EXPORTS)
#define AXISMINT_API  __declspec(dllexport)
#else
#define AXISMINT_API  __declspec(dllimport)
#endif
