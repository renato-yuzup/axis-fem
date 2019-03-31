#ifndef  __SOURCETRACEHINT_HPP
#define  __SOURCETRACEHINT_HPP

#include "Axis.SystemBase.hpp"

namespace axis
{
	namespace foundation
	{
		class AXISSYSTEMBASE_API SourceTraceHint
		{
		private:
			int _hintId;
		public:
			SourceTraceHint(void);
			SourceTraceHint(int hintId);
			virtual ~SourceTraceHint(void);

			bool operator ==(const SourceTraceHint& sth) const;
			bool operator !=(const SourceTraceHint& sth) const;
			bool operator >=(const SourceTraceHint& sth) const;
			bool operator <=(const SourceTraceHint& sth) const;
			bool operator >(const SourceTraceHint& sth) const;
			bool operator <(const SourceTraceHint& sth) const;
		};
	}
}

#endif
