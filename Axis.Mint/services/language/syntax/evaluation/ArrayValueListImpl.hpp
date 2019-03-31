#pragma once
#include "ArrayValueList.hpp"
#include <vector>

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class ArrayValueListImpl : public ArrayValueList
{
public:
	class IteratorLogicImpl : public ArrayValueList::IteratorLogic
	{
	public:
    typedef std::vector<ParameterValue *> param_list;
		IteratorLogicImpl(const param_list::iterator& it);
		virtual ~IteratorLogicImpl(void);
    virtual void Destroy(void) const;
    virtual IteratorLogic& Clone( void ) const;

		virtual const ParameterValue& operator*(void) const;
		virtual const ParameterValue *operator->(void) const;
		virtual IteratorLogic& operator++(void);
		virtual IteratorLogic& operator--(void);
    virtual bool operator ==( const IteratorLogic& other ) const;
	private:
		param_list::iterator _myIterator;
	};

	~ArrayValueListImpl(void);

	virtual bool IsEmpty(void) const;
	virtual int Count(void) const;

	virtual ParameterValue& Get(int pos) const;
	virtual void AddValue(ParameterValue& value);
	virtual void Clear(void);

	virtual Iterator begin( void ) const;
	virtual Iterator end( void ) const;

	virtual ArrayValueList& Clone( void ) const;
private:
	typedef std::vector<ParameterValue *> param_list;
	mutable param_list _list;
};

} } } } } // namespace axis::services::language::syntax::evaluation
