#include "SymbolTable.hpp"
#include <assert.h>
#include "SymbolTable_Pimpl.hpp"
#include "ParseContext.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aapc = axis::application::parsing::core;

namespace {
  static const long AlreadyDefined = -1;

  axis::String pGetSourceString( const axis::String& sourceName)
  {
    if (!sourceName.empty()) return sourceName;
    return axis::String(_T("<unknown>"));
  }

  axis::String pGetLocationString( const axis::String& sourceName, long lineIndex )
  {
    if (!sourceName.empty()) return _T("Ln ") + axis::String::int_parse(lineIndex);
    return axis::String(_T("<unknown>"));
  }

  axis::String pGetDecoratedName( const axis::String& name, const axis::String& typeName )
  {
    return _T("!") + typeName + _T("@@") + name;
  }

  axis::String pResolveTypeName( aapc::SymbolTable::SymbolType type )
  {
    switch(type)
    {
    case aapc::SymbolTable::kNode:
      return _T("@NODE");
      break;
    case aapc::SymbolTable::kNodeDof:
      return _T("@NODE<INIT_DOF>");
      break;
    case aapc::SymbolTable::kNodeSet:
      return _T("@NODE_SET");
      break;
    case aapc::SymbolTable::kElement:
      return _T("@ELEMENT");
      break;
    case aapc::SymbolTable::kElementSet:
      return _T("@ELEMENT_SET");
      break;
    case aapc::SymbolTable::kCurve:
      return _T("@CURVE");
      break;
    case aapc::SymbolTable::kSection:
      return _T("@SECTION");
      break;
    case aapc::SymbolTable::kNodalBoundaryCondition:
      return _T("@BC<NODE>::");
      break;
    case aapc::SymbolTable::kSurfaceDistributedLoad:
      return _T("@SURFACE_LOAD");
      break;
    case aapc::SymbolTable::kEdgeDistributedLoad:
      return _T("@DISTRIBUTED_LOAD");
      break;
    case aapc::SymbolTable::kCustomSymbol:
      return _T("");
      break;
    case aapc::SymbolTable::kAnalysisStep:
      return _T("@ANALYSIS_STEP");
      break;
    case aapc::SymbolTable::kAnalysisSettings:
      return _T("@ANALYSIS_SETTINGS");
      break;
    case aapc::SymbolTable::kOutputFileSettings:
      return _T("@OUTPUT_SETTINGS");
      break;
    default:
      assert(!"Code execution should never reach here.");
    }
    throw "Code execution should never reach here.";
  }
} // namespace


aapc::SymbolTable::SymbolTable(const aapc::ParseContext& context)
{
  pimpl_ = new Pimpl();
  pimpl_->parseContext = &context;
  pimpl_->unresolvedRefCount = 0;
  pimpl_->definedRefCount = 0;
}


aapc::SymbolTable::~SymbolTable(void)
{
  delete pimpl_;
}

void aapc::SymbolTable::AddUnresolvedCustomSymbol( const axis::String& name, 
                                                   const axis::String& typeName )
{
  String decoratedName = pGetDecoratedName(name, typeName);

  // check if symbol is already defined
  if (pimpl_->symbols.find(decoratedName) != pimpl_->symbols.end())
  {
    Pimpl::Symbol& s = *pimpl_->symbols[decoratedName];
    if (s.GetCurrentRefCount() == AlreadyDefined)
    {	// whether this symbol was defined at this or a previous round, it
      // sounds strange to mark it as an unresolved symbol
      throw axis::foundation::InvalidOperationException();
    }

    // if the symbol was last refreshed on a previous round, update it and reset counter
    if (s.GetLastRefreshRound() != pimpl_->parseContext->GetCurrentRoundIndex())
    {
      s.SaveAndResetRefCount();
      s.SetRefreshRound(pimpl_->parseContext->GetCurrentRoundIndex());
    }

    // increment unresolved counter
    s.IncrementRefCount();
  }
  else
  {	// add a new unresolved symbol
    Pimpl::Symbol *s = new Pimpl::Symbol(decoratedName, (int)kCustomSymbol, 
        pGetSourceString(pimpl_->parseContext->GetParseSourceName()), 
        pGetLocationString(pimpl_->parseContext->GetParseSourceName(), 
                           pimpl_->parseContext->GetParseSourceCursorLocation()), 
        pimpl_->parseContext->GetCurrentRoundIndex());
    s->IncrementRefCount();
    pimpl_->symbols[decoratedName] = s;
  }
  pimpl_->unresolvedRefCount++;
}

void aapc::SymbolTable::AddCurrentRoundUnresolvedSymbol( const axis::String& name, SymbolType type )
{
  AddUnresolvedCustomSymbol(name, pResolveTypeName(type));
}

void aapc::SymbolTable::DefineOrRefreshCustomSymbol( const axis::String& name, 
                                                     const axis::String& typeName )
{
  String decoratedName = pGetDecoratedName(name, typeName);

  // check if symbol is already defined
  if (pimpl_->symbols.find(decoratedName) != pimpl_->symbols.end())
  {
    Pimpl::Symbol& s = *pimpl_->symbols[decoratedName];

    if (s.GetLastRefreshRound() != pimpl_->parseContext->GetCurrentRoundIndex())
    {	// symbol must be refreshed to current round
      s.SaveAndResetRefCount();
      s.MarkAsDefined();
      s.SetRefreshRound(pimpl_->parseContext->GetCurrentRoundIndex());
      pimpl_->definedRefCount++;
    }

    if (s.GetCurrentRefCount() != AlreadyDefined)
    {	// reset unresolved counter from symbol
      s.MarkAsDefined();
      pimpl_->definedRefCount++;
    }
  }
  else
  {	// add a new symbol
    Pimpl::Symbol *s = new Pimpl::Symbol(decoratedName, (int)kCustomSymbol, 
        pGetSourceString(pimpl_->parseContext->GetParseSourceName()), 
        pGetLocationString(pimpl_->parseContext->GetParseSourceName(), 
                           pimpl_->parseContext->GetParseSourceCursorLocation()), 
        pimpl_->parseContext->GetCurrentRoundIndex());
    s->MarkAsDefined();
    pimpl_->symbols[decoratedName] = s;
    pimpl_->definedRefCount++;
  }
}

void aapc::SymbolTable::DefineOrRefreshSymbol( const axis::String& name, SymbolType type )
{
  DefineOrRefreshCustomSymbol(name, pResolveTypeName(type));
}

bool aapc::SymbolTable::IsSymbolCurrentRoundDefined( const axis::String& name, 
                                                     const axis::String& typeName ) const
{
  String decoratedName = pGetDecoratedName(name, typeName);
  if (pimpl_->symbols.find(decoratedName) == pimpl_->symbols.end()) return false;
  Pimpl::Symbol *s = pimpl_->symbols[decoratedName];
  return s->GetLastRefreshRound() == pimpl_->parseContext->GetCurrentRoundIndex() &&
         s->GetCurrentRefCount() == AlreadyDefined;
}

bool aapc::SymbolTable::IsSymbolCurrentRoundDefined( const axis::String& name, SymbolType type ) const
{
  return IsSymbolCurrentRoundDefined(name, pResolveTypeName(type));
}

bool aapc::SymbolTable::IsSymbolCurrentRoundUnresolved( const axis::String& name, 
                                                        const axis::String& typeName ) const
{
  String decoratedName = pGetDecoratedName(name, typeName);
  if (pimpl_->symbols.find(decoratedName) == pimpl_->symbols.end()) return false;
  Pimpl::Symbol *s = pimpl_->symbols[decoratedName];
  return s->GetLastRefreshRound() == pimpl_->parseContext->GetCurrentRoundIndex() &&
         s->GetCurrentRefCount() != AlreadyDefined;
}

bool aapc::SymbolTable::IsSymbolCurrentRoundUnresolved( const axis::String& name, 
                                                        SymbolType type ) const
{
  return IsSymbolCurrentRoundUnresolved(name, pResolveTypeName(type));
}

bool aapc::SymbolTable::IsSymbolDefined( const axis::String& name, 
                                         const axis::String& typeName ) const
{
  String decoratedName = pGetDecoratedName(name, typeName);
  bool ok = (pimpl_->symbols.find(decoratedName) != pimpl_->symbols.end());
  if (!ok) return false;
  Pimpl::Symbol *s = pimpl_->symbols[decoratedName];
  return s->GetCurrentRefCount() == AlreadyDefined;
}

bool aapc::SymbolTable::IsSymbolDefined( const axis::String& name, 
                                         SymbolType type ) const
{
  return IsSymbolDefined(name, pResolveTypeName(type));
}

axis::String aapc::SymbolTable::GenerateDecoratedName( SymbolType type )
{
  size_type symbolCount = 0;
  if (pimpl_->autoNames.find((int)type) != pimpl_->autoNames.end())
  {	// we already have a count for auto-generated symbols of this type
    symbolCount = pimpl_->autoNames[type];
  }
  String autoName = _T("!!__AUTO_SYMBOL!") + String::int_parse(symbolCount);
  symbolCount++;
  pimpl_->autoNames[type] = symbolCount;
  return autoName;
}

unsigned long aapc::SymbolTable::GetRoundUnresolvedReferenceCount( void ) const
{
  return pimpl_->unresolvedRefCount;
}

unsigned long aapc::SymbolTable::GetRoundDefinedReferenceCount( void ) const
{
  return pimpl_->definedRefCount;
}
