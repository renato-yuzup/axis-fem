#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class ParseContext;

class AXISPHYSALIS_API SymbolTable
{
public:
  class ReferenceStatistic;
  enum SymbolType
  {
    kNode						          = 0x0000,
    kNodeDof						      = 0x0001,
    kNodeSet						      = 0x0002,
    kElement						      = 0x0003,
    kElementSet					      = 0x0004,
    kCurve						        = 0x0005,
    kSection						      = 0x0006,
    kNodalBoundaryCondition		= 0x0007,
    kSurfaceDistributedLoad		= 0x0041,
    kEdgeDistributedLoad			= 0x0042,
    kCustomSymbol				      = 0xFFF0,
    kAnalysisStep				      = 0xFFFD,
    kOutputFileSettings			  = 0xFFFE,
    kAnalysisSettings			    = 0xFFFF
  };

  SymbolTable(const ParseContext& context);
  ~SymbolTable(void);
  void AddUnresolvedCustomSymbol(const axis::String& name, const axis::String& typeName);
  void AddCurrentRoundUnresolvedSymbol(const axis::String& name, SymbolType type);
  void DefineOrRefreshCustomSymbol(const axis::String& name, const axis::String& typeName);
  void DefineOrRefreshSymbol(const axis::String& name, SymbolType type);
  axis::String GenerateDecoratedName(SymbolType type);
  bool IsSymbolCurrentRoundDefined(const axis::String& name, const axis::String& typeName) const;
  bool IsSymbolCurrentRoundDefined(const axis::String& name, SymbolType type) const;
  bool IsSymbolCurrentRoundUnresolved(const axis::String& name, const axis::String& typeName) const;
  bool IsSymbolCurrentRoundUnresolved(const axis::String& name, SymbolType type) const;
  bool IsSymbolDefined(const axis::String& name, const axis::String& typeName) const;
  bool IsSymbolDefined(const axis::String& name, SymbolType type) const;

  unsigned long GetRoundUnresolvedReferenceCount(void) const;
  unsigned long GetRoundDefinedReferenceCount(void) const;

  friend class ParseContext;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } } // namespace axis::application::parsing::core
