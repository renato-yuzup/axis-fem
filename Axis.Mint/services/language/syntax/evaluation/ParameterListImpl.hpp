#pragma once
#include "ParameterList.hpp"
#include <map>

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class ParameterListImpl : public ParameterList
{
public:
	class IteratorLogicImpl : public ParameterList::IteratorLogic
	{
	public:
    typedef std::map<axis::String, ParameterValue *, axis::StringCompareLessThan> param_list;
		IteratorLogicImpl(param_list::iterator it);
		virtual ~IteratorLogicImpl(void);
		virtual IteratorLogic& Clone( void ) const;

		virtual const ParameterList::Pair& operator*(void) const;
		virtual const ParameterList::Pair *operator->(void) const;
		virtual IteratorLogic& operator++(void);
		virtual IteratorLogic& operator--(void);
		virtual bool operator ==( const IteratorLogic& other ) const;
  private:
    param_list::iterator _myIterator;
    mutable ParameterList::Pair *_myPair;
    mutable bool _isPairValid;
	};

	~ParameterListImpl(void);
	virtual void Destroy(void) const;
  virtual ParameterList& Clone( void ) const;

	virtual bool IsEmpty(void) const;
	virtual int Count(void) const;

	virtual ParameterValue& GetParameterValue(const axis::String& name) const;
	virtual bool IsDeclared(const axis::String& name) const;
	virtual ParameterList& Consume(const axis::String& name);
	virtual ParameterList& AddParameter(const axis::String& name, ParameterValue& value);
	virtual void Clear(void);

	virtual Iterator begin( void ) const;
	virtual Iterator end( void ) const;
private:
	typedef std::map<axis::String, ParameterValue *, axis::StringCompareLessThan> param_list;
	mutable param_list _list;
};

} } } } } // namespace axis::services::language::syntax::evaluation
