#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"
#include "ParameterValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API ArrayValueList
{
public:
	class AXISMINT_API IteratorLogic
	{
	public:
		virtual const ParameterValue& operator*(void) const = 0;
		virtual const ParameterValue *operator->(void) const = 0;
		virtual IteratorLogic& operator++(void) = 0;
		virtual IteratorLogic& operator--(void) = 0;
		virtual IteratorLogic& Clone(void) const = 0;
		virtual bool operator ==(const IteratorLogic& other) const = 0;
		virtual void Destroy(void) const = 0;
	};
	class AXISMINT_API Iterator
	{
	public:
		Iterator(const IteratorLogic& logic);
		Iterator(const Iterator& it);
		Iterator(void);	/* builds an invalid operator */

		const ParameterValue& operator*(void) const;
		const ParameterValue *operator->(void) const;
		Iterator& operator++(void);
		Iterator operator++(int);
		Iterator& operator--(void);
		Iterator operator--(int);

		bool operator ==(const Iterator& it) const;
		bool operator !=(const Iterator& it) const;
		Iterator& operator=(const Iterator& it);
	private:
		void Copy(const Iterator& it);
		IteratorLogic *_logic;
	};

	virtual ~ArrayValueList(void);

	virtual bool IsEmpty(void) const = 0;
	virtual int Count(void) const = 0;

	virtual ParameterValue& Get(int pos) const = 0;
	virtual void AddValue(ParameterValue& value) = 0;
	virtual void Clear(void) = 0;

	virtual Iterator begin(void) const = 0;
	virtual Iterator end(void) const = 0;

	virtual ArrayValueList& Clone(void) const = 0;
	virtual ArrayValueList& operator=(const ArrayValueList& other);
};

} } } } } // namespace axis::services::language::syntax::evaluation
