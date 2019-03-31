#pragma once
#include <map>
#include "SymbolTable.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class SymbolTable::Pimpl
{
public:
  class Symbol 
  {
  public:
    Symbol(const axis::String& decoratedName, int symbolId);
    Symbol(const axis::String& decoratedName, int symbolId, 
           const axis::String& sourcePath, const axis::String& location);
    Symbol(const axis::String& decoratedName, int symbolId, 
           const axis::String& sourcePath, const axis::String& location, 
           int refreshRound);
    Symbol(const axis::String& decoratedName, int symbolId,
           const axis::String& sourcePath, const axis::String& location, 
           int refreshRound, long currentRefCount);
    Symbol(const axis::String& decoratedName, int symbolId, 
           const axis::String& sourcePath, const axis::String& location, 
           int refreshRound, long currentRefCount, long lastRefCount);
    ~Symbol(void);

    axis::String GetDecoratedName(void) const;
    int GetId(void) const;
    axis::String GetSourceString(void) const;
    axis::String GetLocationString(void) const;

    int GetLastRefreshRound(void) const;
    void SetRefreshRound(int roundIndex);

    long GetCurrentRefCount(void) const;
    void SetCurrentRefCount(long count);
    void IncrementRefCount(void);
    void MarkAsDefined(void);

    long GetLastRoundRefCount(void) const;
    void SetLastRoundRefCount(long count);
    void SaveAndResetRefCount(void);
  private:
    const axis::String decoratedName_;
    const int symbolId_;
    int lastRefreshRound_;
    const axis::String source_;
    const axis::String location_;
    long lastRefCount_;
    long currentRefCount_;
  };

  typedef std::map<axis::String, Symbol *> symbol_map;

  Pimpl(void);
  ~Pimpl(void);

  symbol_map symbols;
  typedef std::map<int,size_type> autoname_map;
  autoname_map autoNames;
  const ParseContext *parseContext;
  unsigned long unresolvedRefCount;
  unsigned long definedRefCount;
};

} } } } // namespace axis::application::parsing::core
