#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace io
		{
			class AXISSYSTEMBASE_API DirectoryNavigator
			{
			protected:
				DirectoryNavigator(void);
			public:
				static DirectoryNavigator& Create(const axis::String& path, bool isRecursive);
				static DirectoryNavigator& Create(const axis::String& path);

				virtual ~DirectoryNavigator(void);

				virtual void Destroy(void) const = 0;

				virtual axis::String GetFile(void) const = 0;
				virtual axis::String GetFileName(void) const = 0;
				virtual axis::String GetFileExtension(void) const = 0;

				virtual bool HasNext(void) const = 0;

				virtual void GoNext(void) const = 0;
			};
		}
	}
}

