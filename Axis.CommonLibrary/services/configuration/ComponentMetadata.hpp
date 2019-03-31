#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis
{
	namespace services
	{
		namespace configuration
		{
			class AXISCOMMONLIBRARY_API ComponentMetadata
			{
			private:
				char *_featurePath;
				char *_featureName;

				void CopyStr(char **ptr, const char *source);
			public:
				ComponentMetadata(const char *featurePath, const char *shortName);
				~ComponentMetadata(void);

				const char *FeaturePath(void) const;
				const char *ShortName(void) const;
			};
			
		}
	}
}

