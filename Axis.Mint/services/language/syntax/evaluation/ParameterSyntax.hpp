#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace services { namespace language { namespace syntax { namespace evaluation {

class ParameterValue;
class ParameterList;

/**
 * Provides services to aid in parameter checking.
**/
class AXISMINT_API ParameterSyntax
{
public:
  ~ParameterSyntax(void);

  static bool IsBoolean(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static bool IsId(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static bool IsString(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static bool IsIdOrString(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static bool IsNumeric(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static bool IsParameterEnumeration(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);

  static bool DeclaresBoolean(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static bool DeclaresId(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static bool DeclaresString(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static bool DeclaresIdOrString(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static bool DeclaresNumeric(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static bool DeclaresParameterEnumeration(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);

  static bool GetBooleanValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static bool GetBooleanValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName, bool defaultValue);
  static axis::String GetStringValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static axis::String GetStringValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName, const axis::String& defaultValue);
  static axis::String GetIdValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static axis::String GetIdValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName, const axis::String& defaultValue);
  static axis::String GetIdOrStringValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static axis::String GetIdOrStringValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName, const axis::String& defaultValue);
  static long GetIntegerValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static long GetIntegerValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName, long defaultValue);
  static real GetRealValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName);
  static real GetRealValue(
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    const axis::String& paramName, real defaultValue);

  static bool ToBoolean(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static axis::String ToString(
    const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static real ToReal(const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
  static long ToInteger(const axis::services::language::syntax::evaluation::ParameterValue& paramVal);
private:
  ParameterSyntax(void);
};

} } } } } // namespace axis::services::language::syntax::evaluation
