#pragma once
#include "SourceHintCollection.hpp"
#include <set>

namespace axis
{
	namespace foundation
	{
		class SourceHintSet : public SourceHintCollection
		{
		private:
			typedef std::set<const SourceTraceHint *> hint_set;
			hint_set _items;
		public:
			class SourceHintSetVisitor : public SourceHintCollection::Visitor
			{
			private:
				typedef std::set<const SourceTraceHint *> hint_set;
				hint_set::iterator _begin;
				hint_set::iterator _current;
				hint_set::iterator _end;
			public:
				SourceHintSetVisitor(hint_set::iterator begin, hint_set::iterator end, hint_set::iterator current);
				virtual ~SourceHintSetVisitor(void);

				virtual const SourceTraceHint& GetItem(void) const;
				virtual bool HasNext(void) const;
				virtual void GoNext(void);
				virtual void Reset(void);
			};

			virtual void Add( const SourceTraceHint& hint );

			virtual void Remove( const SourceTraceHint& hint );

			virtual void Clear( void );

			virtual bool IsEmpty( void ) const;

			virtual bool Contains( const SourceTraceHint& hint ) const;

			virtual Visitor& GetVisitor( void ) const;
		};
	}
}