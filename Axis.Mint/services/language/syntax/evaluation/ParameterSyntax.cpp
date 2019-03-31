#include "ParameterSyntax.hpp"
#include "ParameterValue.hpp"
#include "AtomicValue.hpp"
#include "ArrayValue.hpp"
#include "ParameterList.hpp"
#include "NumberValue.hpp"

namespace aslse = axis::services::language::syntax::evaluation;

aslse::ParameterSyntax::~ParameterSyntax(void)
{
  // nothing to do here
}

bool aslse::ParameterSyntax::IsBoolean( const aslse::ParameterValue& paramVal )
{
  if (!paramVal.IsAtomic()) return false;
  const AtomicValue& val = static_cast<const AtomicValue&>(paramVal);
  if (!(val.IsId() || val.IsString())) return false;
  String paramValueStr = val.ToString();
  paramValueStr.to_lower_case();
  if (paramValueStr == _T("true") || paramValueStr == _T("false") || 
      paramValueStr == _T("yes") || paramValueStr == _T("no") ||
      paramValueStr == _T("on") || paramValueStr == _T("off"))
  {
    return true;
  }
  return false;
}

bool aslse::ParameterSyntax::IsId( const aslse::ParameterValue& paramVal )
{
  if (!paramVal.IsAtomic()) return false;
  const AtomicValue& val = static_cast<const AtomicValue&>(paramVal);
  return val.IsId();
}

bool aslse::ParameterSyntax::IsString( const aslse::ParameterValue& paramVal )
{
  if (!paramVal.IsAtomic()) return false;
  const AtomicValue& val = static_cast<const AtomicValue&>(paramVal);
  return val.IsString();
}

bool aslse::ParameterSyntax::IsIdOrString( const aslse::ParameterValue& paramVal )
{
  return IsId(paramVal) || IsString(paramVal);
}

bool aslse::ParameterSyntax::IsNumeric( const aslse::ParameterValue& paramVal )
{
  if (!paramVal.IsAtomic()) return false;
  const AtomicValue& val = static_cast<const AtomicValue&>(paramVal);
  return val.IsNumeric();
}

bool aslse::ParameterSyntax::IsParameterEnumeration( const aslse::ParameterValue& paramVal )
{
  if (!paramVal.IsAtomic()) return false;
  const AtomicValue& val = static_cast<const AtomicValue&>(paramVal);
  if (!val.IsArray()) return false;
  const ArrayValue& arrayValue = static_cast<const ArrayValue&>(paramVal);
  try
  {
    ParameterList& p = ParameterList::FromParameterArray(arrayValue);
    p.Destroy();
  }
  catch (...)
  {
    return false;
  }
  return true;
}

bool aslse::ParameterSyntax::DeclaresBoolean( const aslse::ParameterList& paramList, 
                                              const axis::String& paramName )
{
  if (!paramList.IsDeclared(paramName)) return false;
  const ParameterValue& paramVal = paramList.GetParameterValue(paramName);
  return IsBoolean(paramVal);
}

bool aslse::ParameterSyntax::DeclaresId( const aslse::ParameterList& paramList, 
                                         const axis::String& paramName )
{
  if (!paramList.IsDeclared(paramName)) return false;
  const ParameterValue& paramVal = paramList.GetParameterValue(paramName);
  return IsId(paramVal);
}

bool aslse::ParameterSyntax::DeclaresString( const aslse::ParameterList& paramList, 
                                             const axis::String& paramName )
{
  if (!paramList.IsDeclared(paramName)) return false;
  const ParameterValue& paramVal = paramList.GetParameterValue(paramName);
  return IsString(paramVal);
}

bool aslse::ParameterSyntax::DeclaresIdOrString( const aslse::ParameterList& paramList, 
                                                 const axis::String& paramName )
{
  if (!paramList.IsDeclared(paramName)) return false;
  const ParameterValue& paramVal = paramList.GetParameterValue(paramName);
  return IsIdOrString(paramVal);
}

bool aslse::ParameterSyntax::DeclaresNumeric( const aslse::ParameterList& paramList, 
                                              const axis::String& paramName )
{
  if (!paramList.IsDeclared(paramName)) return false;
  const ParameterValue& paramVal = paramList.GetParameterValue(paramName);
  return IsNumeric(paramVal);
}

bool aslse::ParameterSyntax::DeclaresParameterEnumeration( const aslse::ParameterList& paramList, 
                                                           const axis::String& paramName )
{
  if (!paramList.IsDeclared(paramName)) return false;
  const ParameterValue& paramVal = paramList.GetParameterValue(paramName);
  return IsParameterEnumeration(paramVal);
}

bool aslse::ParameterSyntax::GetBooleanValue( const aslse::ParameterList& paramList, 
                                              const axis::String& paramName )
{
  const AtomicValue& paramVal = 
    static_cast<const AtomicValue&>(paramList.GetParameterValue(paramName));  
  String paramValueStr = paramVal.ToString();
  paramValueStr.to_lower_case();
  if (paramValueStr == _T("true") || paramValueStr == _T("yes") || paramValueStr == _T("on"))
  {
    return true;
  }
  return false;
}

bool aslse::ParameterSyntax::GetBooleanValue( const aslse::ParameterList& paramList, 
                                              const axis::String& paramName, bool defaultValue )
{
  if (!paramList.IsDeclared(paramName)) return defaultValue;
  return GetBooleanValue(paramList, paramName);
}

axis::String aslse::ParameterSyntax::GetStringValue( const aslse::ParameterList& paramList, 
                                                     const axis::String& paramName )
{
  const AtomicValue& paramVal = static_cast<const AtomicValue&>(paramList.GetParameterValue(paramName));  
  return paramVal.ToString();
}

axis::String aslse::ParameterSyntax::GetStringValue( const aslse::ParameterList& paramList, 
                                                     const axis::String& paramName, 
                                                     const axis::String& defaultValue )
{
  if (!paramList.IsDeclared(paramName)) return defaultValue;
  return GetStringValue(paramList, paramName);
}

axis::String aslse::ParameterSyntax::GetIdValue( const aslse::ParameterList& paramList, 
                                                 const axis::String& paramName )
{
  const AtomicValue& paramVal = static_cast<const AtomicValue&>(paramList.GetParameterValue(paramName));  
  return paramVal.ToString();
}

axis::String aslse::ParameterSyntax::GetIdValue( const aslse::ParameterList& paramList, 
                                                 const axis::String& paramName, 
                                                 const axis::String& defaultValue )
{
  if (!paramList.IsDeclared(paramName)) return defaultValue;
  return GetIdValue(paramList, paramName);
}

axis::String aslse::ParameterSyntax::GetIdOrStringValue( const aslse::ParameterList& paramList, 
                                                         const axis::String& paramName )
{
  const AtomicValue& paramVal = static_cast<const AtomicValue&>(paramList.GetParameterValue(paramName));  
  return paramVal.ToString();
}

axis::String aslse::ParameterSyntax::GetIdOrStringValue( const aslse::ParameterList& paramList, 
                                                         const axis::String& paramName, 
                                                         const axis::String& defaultValue )
{
  if (!paramList.IsDeclared(paramName)) return defaultValue;
  return GetIdOrStringValue(paramList, paramName);
}

long aslse::ParameterSyntax::GetIntegerValue( const aslse::ParameterList& paramList, 
                                              const axis::String& paramName )
{
  const AtomicValue& paramVal = static_cast<const AtomicValue&>(paramList.GetParameterValue(paramName));  
  const NumberValue& numberVal = static_cast<const NumberValue&>(paramVal);
  return numberVal.GetLong();
}

long aslse::ParameterSyntax::GetIntegerValue( const aslse::ParameterList& paramList, 
                                              const axis::String& paramName, 
                                              long defaultValue )
{
  if (!paramList.IsDeclared(paramName)) return defaultValue;
  return GetIntegerValue(paramList, paramName);
}

real aslse::ParameterSyntax::GetRealValue( const aslse::ParameterList& paramList, 
                                           const axis::String& paramName )
{
  const AtomicValue& paramVal = static_cast<const AtomicValue&>(paramList.GetParameterValue(paramName));  
  const NumberValue& numberVal = static_cast<const NumberValue&>(paramVal);
  return numberVal.GetDouble();
}

real aslse::ParameterSyntax::GetRealValue( const aslse::ParameterList& paramList, 
                                           const axis::String& paramName, 
                                           real defaultValue )
{
  if (!paramList.IsDeclared(paramName)) return defaultValue;
  return GetRealValue(paramList, paramName);
}

bool aslse::ParameterSyntax::ToBoolean( const aslse::ParameterValue& paramVal )
{
  const aslse::AtomicValue& val = static_cast<const aslse::AtomicValue&>(paramVal);
  axis::String s = val.ToString();
  s.to_lower_case();
  return (s == _T("true") || s == _T("on") || s == _T("yes"));
}

axis::String aslse::ParameterSyntax::ToString( const aslse::ParameterValue& paramVal )
{
  return paramVal.ToString();
}

real aslse::ParameterSyntax::ToReal( const aslse::ParameterValue& paramVal )
{
  const aslse::NumberValue& val = static_cast<const aslse::NumberValue&>(paramVal);
  return val.GetDouble();
}

long aslse::ParameterSyntax::ToInteger( const aslse::ParameterValue& paramVal )
{
  const aslse::NumberValue& val = static_cast<const aslse::NumberValue&>(paramVal);
  return val.GetLong();
}
