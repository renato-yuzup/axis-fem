#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis { namespace domain { namespace collections {

class AXISCOMMONLIBRARY_API FlagCollection
{
public:
  FlagCollection(void);
	~FlagCollection(void);

	void Add(const axis::String& flag);
	void Remove(const axis::String& flag);
	bool IsDefined(const axis::String& flag) const;
	void Clear(void);
	unsigned int Count(void) const;

	FlagCollection& Clone(void) const;
	void Destroy(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

		
} } } // namespace axis::domain::collections
