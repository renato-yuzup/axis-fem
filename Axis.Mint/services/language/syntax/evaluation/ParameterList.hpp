#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"
#include "ParameterValue.hpp"
#include "ArrayValue.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class AXISMINT_API ParameterList
{
public:
	class AXISMINT_API Pair
	{
	public:
		Pair(const axis::String& name, ParameterValue *value);
		Pair(void);
		Pair(const Pair& other);
		Pair& operator =(const Pair& pair);
		bool operator ==(const Pair& pair) const;
		bool operator !=(const Pair& pair) const;

		/* Public members */
		axis::String Name;
		ParameterValue *Value;
	};

	class AXISMINT_API IteratorLogic
	{
	public:
		virtual const Pair& operator*(void) const = 0;
		virtual const Pair *operator->(void) const = 0;
		virtual IteratorLogic& operator++(void) = 0;
		virtual IteratorLogic& operator--(void) = 0;
		virtual IteratorLogic& Clone(void) const = 0;

		virtual bool operator ==(const IteratorLogic& other) const = 0;
	};
	class AXISMINT_API Iterator
	{
	public:
		Iterator(const IteratorLogic& logic);
		Iterator(const Iterator& it);
		Iterator(void);	/* builds an invalid operator */

		const Pair& operator*(void) const;
		const Pair *operator->(void) const;
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

	virtual ~ParameterList(void);
	virtual void Destroy(void) const = 0;

	virtual bool IsEmpty(void) const = 0;
	virtual int Count(void) const = 0;

	virtual ParameterValue& GetParameterValue(const axis::String& name) const = 0;
	virtual bool IsDeclared(const axis::String& name) const = 0;
	virtual ParameterList& Consume(const axis::String& name) = 0;
	virtual ParameterList& AddParameter(const axis::String& name, ParameterValue& value) = 0;
	virtual void Clear(void) = 0;
						
	virtual Iterator begin(void) const = 0;
	virtual Iterator end(void) const = 0;

	virtual ParameterList& Clone(void) const = 0;
	virtual ParameterList& operator=(const ParameterList& other);

	static ParameterList& Create(void);
	static const ParameterList& Empty;
	static ParameterList& FromParameterArray(
    const axis::services::language::syntax::evaluation::ArrayValue& arrayValue);
};

} } } } } // namespace axis::services::language::syntax::evaluation
