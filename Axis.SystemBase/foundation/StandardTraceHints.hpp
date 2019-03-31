#ifndef __STANDARDTRACEHINTS_HPP
#define __STANDARDTRACEHINTS_HPP
#include "SourceTraceHint.hpp"
#include "Axis.SystemBase.hpp"

namespace axis
{
	namespace foundation
	{
		class AXISSYSTEMBASE_API StandardTraceHints
		{
		private:
			StandardTraceHints(void);	// cannot instantiate
		public:
			~StandardTraceHints(void);

			static const axis::foundation::SourceTraceHint& AnalysisBlockReaderLogic;
			static const axis::foundation::SourceTraceHint& ModuleManagerControl;
			static const axis::foundation::SourceTraceHint& InputStreamFormatter;
			static const axis::foundation::SourceTraceHint& PreProcessorControl;

		};
	}
}

#endif
