#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API ResultCollectionBlockSyntax
{
private:
	ResultCollectionBlockSyntax(void);
public:
  static const axis::String::char_type * FileNameParameterName;
  static const axis::String::char_type * FileFormatParameterName;
  static const axis::String::char_type * AppendParameterName;
  static const axis::String::char_type * FormatArgumentsParameterName;
  static const axis::String::char_type * BlockName;

	friend class AxisInputLanguage;
};	

} } } // namespace axis::foundation::definitions
