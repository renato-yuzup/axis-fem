#pragma once
#include "Axis.SystemBase.hpp"
#include "SourceTraceHint.hpp"

namespace axis
{
	namespace foundation
	{
		class AXISSYSTEMBASE_API SourceHintCollection
		{
		public:
			class AXISSYSTEMBASE_API Visitor
			{
			public:
				virtual ~Visitor(void);

				virtual const SourceTraceHint& GetItem(void) const = 0;
				virtual bool HasNext(void) const = 0;
				virtual void GoNext(void) = 0;
				virtual void Reset(void) = 0;
			};

			virtual ~SourceHintCollection(void);

			virtual void Add(const SourceTraceHint& hint) = 0;
			virtual void Remove(const SourceTraceHint& hint) = 0;
			virtual void Clear(void) = 0;

			virtual bool IsEmpty(void) const = 0;
			virtual bool Contains(const SourceTraceHint& hint) const = 0;

			virtual Visitor& GetVisitor(void) const = 0;
		};
	}
}