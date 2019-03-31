#pragma once

#if defined(AXISORANGE_EXPORTS)
#define AXISORANGE_API  __declspec(dllexport)
#else
#define AXISORANGE_API  __declspec(dllimport)
#endif
